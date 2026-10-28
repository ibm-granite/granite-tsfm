import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, dim=-1, std_min=1e-5, max_val=100, use_sinh=False):
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.max_val = max_val
        self.use_sinh = use_sinh

    def fit_transform(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            mean, std = self.get_statistics(x, mask)
            return self.transform(x, mean=mean, std=std), mean, std

    def transform(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            x = (x - mean) / std
            if self.use_sinh:
                x = torch.asinh(x)
            return x

    def inverse_transform(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if self.use_sinh:
                x = torch.sinh(x)
            if x.ndim != mean.ndim:
                x = x * std.unsqueeze(1) + mean.unsqueeze(1)
            else:
                x = x * std + mean

            return x

    def get_statistics(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            x_mean = x.mean(dim=self.dim, keepdim=True)
            x_std = x.std(dim=self.dim, keepdim=True)
            x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
        else:
            mask = mask.bool()
            unmask = (~mask).float()
            count = unmask.sum(dim=self.dim, keepdim=True).clamp(min=1)  # avoid division by zero
            x_mean = (x * unmask).sum(dim=self.dim, keepdim=True) / count
            x_std = (((x - x_mean) * unmask) ** 2).sum(dim=self.dim, keepdim=True) / count
            x_std = x_std.sqrt()
            x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
        return x_mean, x_std


class CausalRevIN(nn.Module):
    def __init__(self, dim=-1, std_min=1e-5, max_val=100):
        """
        Causal RevIN implementation to enable parallel predictions during training of FlowState

        :param eps: a value added for numerical stability
        :param with_missing (bool): whether contiguous patch masking (CPM) is used or not, interpreting nans as missing values
        """
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.max_val = max_val

    def fit_transform(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.get_statistics(x, mask)
        return self.transform(x, mean=mean, std=std), mean, std

    def transform(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - mean) / std, min=-self.max_val, max=self.max_val)

    def inverse_transform(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x * std + mean
        elif x.ndim == 3:
            return x * std.unsqueeze(-1) + mean.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid input dimension: {x.shape}")

    def get_statistics(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            n = torch.cumsum(1 - mask.float(), dim=1)
            n = torch.where(n == 0, 1.0, n)
        else:
            n = torch.arange(1, x.shape[1] + 1, device=x.device)
        x_mean = (torch.cumsum(x, dim=1) / n).detach()
        mask = 1 - mask.float() if mask is not None else 1
        x_std = torch.sqrt(torch.cumsum(((x - x_mean) * mask) ** 2, 1) / n).detach()
        x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
        return x_mean, x_std
