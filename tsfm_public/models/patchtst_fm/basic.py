from typing import Optional, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_attn_mask(query_pad: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """
    Build an additive attention mask of shape (B, Q, K) from
    query/key padding masks.

    Args:
        query_pad: (B, Q) bool or 0/1 tensor. 1/True = padded query position.
        key_pad:   (B, K) bool or 0/1 tensor. 1/True = padded key position.

    Returns:
        attn_mask: (B, Q, K) float tensor, where masked positions are -inf
                   and valid positions are 0.0 (for use with SDPA).
    """
    # Ensure boolean
    q_pad = query_pad.bool()  # (B, Q)
    k_pad = key_pad.bool()  # (B, K)

    # A position (q, k) is invalid if *either* the query or key is padded
    # Shape: (B, Q, K)
    pad = q_pad.unsqueeze(-1) | k_pad.unsqueeze(-2)

    # Build float mask with -inf on padded positions, 0 elsewhere
    attn_mask = torch.zeros_like(pad, dtype=torch.float32)
    attn_mask.masked_fill_(pad, float("-inf"))

    return attn_mask


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=256,
        num_hidden_layers=1,
        dropout=0,
        norm=False,
        activation=nn.GELU(approximate="tanh"),
        output_activation=nn.Identity(),
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        # layers.append(norm_layer(hidden_dim) if norm else nn.Identity())
        layers.append(activation)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(norm_layer(hidden_dim) if norm else nn.Identity())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Dropout(dropout))
        layers.append(norm_layer(hidden_dim) if norm else nn.Identity())
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(output_activation)
        self.layers = nn.Sequential(*layers)
        # self.init_weights()

    def forward(self, x):
        return self.layers(x)


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=384, dropout=0):
        super().__init__()
        hidden_dim = round(hidden_dim * 2 / 3)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.activation(self.fc2(x))
        return self.dropout(self.fc3(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - x                          when output_attentions=False  (normal path)
            - (x, attn_weights)          when output_attentions=True
              attn_weights shape (3-D input): (B, num_heads, N, N)
              The weights are computed in fp32 via an explicit softmax; the main
              forward pass always uses the fused SDPA kernel unchanged.
        """
        if x.ndim == 3:
            B, N, C = x.shape
            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            )  # 3 x B x num_heads x N (num_patches) x head_dim
            q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
            # with sdpa_kernel(SDPBackend.MATH):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )  # (B, num_heads, N, head_dim)
            x = x.transpose(1, 2).reshape(B, N, C)
        elif x.ndim == 4:
            B, M, N, C = x.shape
            qkv = self.qkv(x).reshape(B, M, N, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv.unbind(0)  # (B, num_heads, M, N, head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
            # print('q', q.shape, 'k', k.shape, 'v', v.shape, 'attn_mask', attn_mask.shape if attn_mask is not None else "None")
            # with sdpa_kernel(SDPBackend.MATH):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
            )
            x = x.permute(0, 2, 3, 1, 4).reshape(B, M, N, C)
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")
        x = self.proj(x)
        x = self.proj_drop(x)

        if not output_attentions:
            return x

        # The following is for attention visualization
        assert not self.training, (
            "output_attentions must not be used during training "
            "(pass --record_attention only at eval/inference time)"
        )
        with torch.no_grad():
            q_fp32 = q.float()
            k_fp32 = k.float()
            if q_fp32.ndim == 5:
                # 4-D input path: q/k are (B, H, M, N, Hd) — collapse M by mean
                q_fp32 = q_fp32.mean(dim=2)  # (B, H, N, Hd)
                k_fp32 = k_fp32.mean(dim=2)
            # Build score matrix (B, H, N, N)
            scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                mask_fp32 = attn_mask.float()
                # Identify rows that are all -inf (i.e. fully padded query rows)
                # and replace those rows with 0 so softmax gives uniform weights.
                all_masked_rows = (mask_fp32 <= -1e4).all(dim=-1, keepdim=True)  # (B, 1, N, 1)
                safe_mask = mask_fp32.masked_fill(all_masked_rows.expand_as(mask_fp32), 0.0)
                scores = scores + safe_mask
            attn_weights = torch.softmax(scores, dim=-1)  # (B, H, N, N)
        return x, attn_weights


class CrossAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,  # dim of x
        kv_dim: Optional[int] = None,  # dim of m (defaults to q_dim)
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else q_dim
        assert q_dim % num_heads == 0, "q_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads

        self.q = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)  # produce k and v in the SAME head dim as q
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,  # (B, Nq, q_dim)
        m: torch.Tensor,  # (B, Nk, kv_dim)
        attn_mask: Optional[torch.Tensor] = None,  # broadcastable to (B, num_heads, Nq, Nk) or (Nq, Nk)
        is_causal: bool = False,
    ) -> torch.Tensor:
        if x.ndim == 3:
            B, Nq, Cq = x.shape
            _, Nk, _ = m.shape
            q = self.q(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, Nq, Hd)
            kv = self.kv(m).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)  # (B, H, Nk, Hd)
            q, k = self.q_norm(q), self.k_norm(k)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )  # (B, H, Nq, Hd)
            x = x.transpose(1, 2).reshape(B, Nq, Cq)  # back to (B, Nq, q_dim)
        elif x.ndim == 4:
            B, M, Nq, Cq = x.shape
            _, Nk, _ = m.shape
            q = self.q(x).reshape(B, M, Nq, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, H, M, Nq, Hd)
            kv = self.kv(m).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)  # (B, H, Nk, Hd)
            q, k = self.q_norm(q), self.k_norm(k)
            x = F.scaled_dot_product_attention(
                q,
                k.unsqueeze(2),
                v.unsqueeze(2),
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )  # (B, H, M, Nq, Hd)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, M, Nq, Cq)
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    """
    A standard Transformer block.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        norm_first=True,
        norm_layer=nn.LayerNorm,
        mlp_type="mlp",
        qkv_bias=True,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(d_model, num_heads, qkv_bias=qkv_bias, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(d_model, d_model, hidden_dim=int(mlp_ratio * d_model), dropout=dropout)
        elif mlp_type == "mlp":
            self.mlp = MLP(
                in_dim=d_model,
                out_dim=d_model,
                hidden_dim=int(mlp_ratio * d_model),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported MLP type: {mlp_type}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, output_attentions: bool = False):
        if self.norm_first:
            attn_out = self.attn(self.norm1(x), attn_mask, output_attentions=output_attentions)
            if output_attentions:
                attn_out, attn_weights = attn_out
            x = x + attn_out
            x = x + self.dropout(self.mlp(self.norm2(x)))
        else:
            attn_out = self.attn(x, attn_mask, output_attentions=output_attentions)
            if output_attentions:
                attn_out, attn_weights = attn_out
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.dropout(self.mlp(x)))
        if output_attentions:
            return x, attn_weights
        return x


class ConformerBlock(nn.Module):
    """
    Conformer block combining Transformer attention with convolution.
    Architecture: FFN(half) -> Attention -> Convolution -> FFN(half)
    """

    def __init__(
        self,
        d_model,
        num_heads,
        conv_kernel_size=31,
        is_causal=False,
        mlp_ratio=4.0,
        dropout=0.1,
        norm_first=True,
        norm_layer=nn.LayerNorm,
        mlp_type="mlp",
        conv_block_type="default",
        ssm_state_dim=16,
        qkv_bias=True,
        sublayer_gating=False,
        sublayer_gating_init_value=1e-4,
    ):
        """ 
        conv_block_type = {"default", "ssm", "esp" }
        """
        super().__init__()
        self.norm_first = norm_first
        self.conv_block_type = conv_block_type
        self.is_causal = is_causal   # will make the convo block causal (left padding only), allows for even-numbered size, too
        self.sublayer_gating = sublayer_gating   # this might get removed in the future
        self.sublayer_gating_init_value = sublayer_gating_init_value

        # residual gate weight
        if self.sublayer_gating:
            if not self.norm_first:
                raise ValueError("Do not use sublayer_gating=True with norm_first=False.")
            self.bias_ffn1_gate_alpha  = nn.Parameter(torch.tensor(self.sublayer_gating_init_value))
            self.bias_attn_gate_alpha  = nn.Parameter(torch.tensor(self.sublayer_gating_init_value))
            self.bias_conv_gate_alpha  = nn.Parameter(torch.tensor(self.sublayer_gating_init_value))
            self.bias_ffn2_gate_alpha  = nn.Parameter(torch.tensor(self.sublayer_gating_init_value))

        # First half-step FFN (expansion factor 0.5)
        self.norm_ffn1 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        if mlp_type == "swiglu":
            self.ffn1 = SwiGLU(d_model, d_model, hidden_dim=int(mlp_ratio * d_model // 2), dropout=dropout)
        else:
            self.ffn1 = MLP(
                in_dim=d_model,
                out_dim=d_model,
                hidden_dim=int(mlp_ratio * d_model // 2),
                dropout=dropout,
            )
        
        # Multi-head self-attention
        self.norm_attn = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(d_model, num_heads, qkv_bias=qkv_bias, attn_drop=dropout, proj_drop=dropout)
        
        self.norm_conv = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        self.conv_block_type == "default"
        # Depthwise convolution
        if not is_causal:
            self.conv = nn.Sequential(
                #nn.Conv1d(
                nn.Conv2d(
                    d_model, 
                    d_model, 
                    #kernel_size=conv_kernel_size,
                    kernel_size=(1, conv_kernel_size),
                    padding=(0,conv_kernel_size // 2),
                    dilation=(1,1),
                    stride=(1,1),
                    #padding=conv_kernel_size // 2,
                    groups=d_model,  # Depthwise
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            to_be_padded = conv_kernel_size - 1
            self.conv = nn.Sequential(
                nn.ConstantPad1d((to_be_padded, 0), 0.0),
                nn.Conv1d(
                    d_model, 
                    d_model, 
                    kernel_size=conv_kernel_size,
                    padding=0,
                    groups=d_model,  # Depthwise
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        
        # Second half-step FFN
        self.norm_ffn2 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        if mlp_type == "swiglu":
            self.ffn2 = SwiGLU(d_model, d_model, hidden_dim=int(mlp_ratio * d_model // 2), dropout=dropout)
        else:
            self.ffn2 = MLP(
                in_dim=d_model,
                out_dim=d_model,
                hidden_dim=int(mlp_ratio * d_model // 2),
                dropout=dropout,
            )
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attn_mask=None, output_attentions: bool = False):
        # x shape: (B, N, C) where N is sequence length, C is d_model

        if self.norm_first:
            # First half-step FFN
            if not self.sublayer_gating:
                x = x + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))
            else:
                x = x + self.bias_ffn1_gate_alpha * 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))

            # Multi-head attention
            attn_out = self.attn(self.norm_attn(x), attn_mask, output_attentions=output_attentions)
            if output_attentions:
                attn_out, attn_weights = attn_out
            if not self.sublayer_gating:
                x = x + attn_out
            else:
                x = x + self.bias_attn_gate_alpha * attn_out

            # Convolution module
            residual = x
            x = self.norm_conv(x)
            if self.conv_block_type != "default":
                x = self.conv(x)   # transpose done within corresp. class
            else:
                # Conv1d expects (B, C, N)
                x = x.transpose(1, 2).unsqueeze(2)
                x = self.conv(x).squeeze(2)
                x = x.transpose(1, 2)
            if not self.sublayer_gating:
                x = residual + self.dropout(x)
            else:
                x = residual + self.bias_conv_gate_alpha * self.dropout(x)

            # Second half-step FFN
            if not self.sublayer_gating:
                # default
                x = x + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
            else:
                # weight
                x = x + self.bias_ffn2_gate_alpha * 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
        else:
            # Post-norm variant
            x = self.norm_ffn1(x + 0.5 * self.dropout(self.ffn1(x)))
            attn_out = self.attn(x, attn_mask, output_attentions=output_attentions)
            if output_attentions:
                attn_out, attn_weights = attn_out
            x = self.norm_attn(x + attn_out)

            residual = x
            if self.use_ssm:
                x = self.conv(x)
            else:
                x = x.transpose(1, 2)
                x = self.conv(x)
                x = x.transpose(1, 2)
            x = self.norm_conv(residual + self.dropout(x))

            x = self.norm_ffn2(x + 0.5 * self.dropout(self.ffn2(x)))

        if output_attentions:
            return x, attn_weights
        return x



class TransformerBlockCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_cond=None,
        mlp_ratio=4.0,
        dropout=0.1,
        norm_first=True,
        norm_layer=nn.LayerNorm,
        mlp_type="mlp",
    ):
        super().__init__()
        d_cond = d_cond if d_cond is not None else d_model
        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        self.attn = CrossAttention(
            d_model,
            d_cond,
            num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(d_model, d_model, hidden_dim=int(mlp_ratio * d_model), dropout=dropout)
        elif mlp_type == "mlp":
            self.mlp = MLP(
                in_dim=d_model,
                out_dim=d_model,
                hidden_dim=int(mlp_ratio * d_model),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported MLP type: {mlp_type}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, m, attn_mask=None):
        if self.norm_first:
            x = x + self.attn(self.norm1(x), m, attn_mask)
            x = x + self.dropout(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(x + self.attn(x, m, attn_mask))
            x = self.norm2(x + self.dropout(self.mlp(x)))
        return x
