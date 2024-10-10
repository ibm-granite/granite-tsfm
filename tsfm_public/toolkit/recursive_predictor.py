# Copyright contributors to the TSFM project
#
"""Recursive prediction model wrapper"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
)


@dataclass
class RecursivePredictorOutput(ModelOutput):
    """
    Output type of [`RecursivePredictorOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the prediction head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None


class RecursivePredictorConfig(PretrainedConfig):
    model_type = "recursivepredictor"
    """
    RecursivePredictorConfig

    Args:
        model (PreTrainedModel): Model to load for recursive predictions
        requested_prediction_length (int): Total prediction length
        model_prediction_length (int): prediction length of the model
        loss (str): loss to report
    """

    def __init__(
        self,
        model: PreTrainedModel,
        requested_prediction_length: int,
        model_prediction_length: int,
        loss: str,
        **kwargs,
    ):
        self.model = model
        self.requested_prediction_length = requested_prediction_length
        self.model_prediction_length = model_prediction_length
        self.loss = loss
        super().__init__(**kwargs)


class RecursivePredictorPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = RecursivePredictorConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        pass


class RecursivePredictor(RecursivePredictorPreTrainedModel):
    def __init__(self, config: RecursivePredictorConfig):
        super().__init__(config)
        self.model = config.model
        self.requested_prediction_length = config.requested_prediction_length
        self.model_prediction_length = config.model_prediction_length
        self.use_return_dict = config.use_return_dict
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "mae":
            self.loss = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("Invalid loss function: Allowed values: mse and mae")

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None,
    ) -> RecursivePredictorOutput:
        """
        Predict future points given an input sequence, using a recursive strategy.

        Assumptions: The model passed as part of the initialization, should support the following:
         - the signature of the forward method should allow the following arguments: past_values,
         future_vales, past_observed_mask, future_observed_mask, freq_token.
         - the model should have a config attribute prediction_channel_indices which indicates the
         indices in the past and future_value tensors which correspond to the channels we wish to predict
         - if future_values is provided, it must be of shape compatible with the requested_prediction_length

        Args:
            past_values (torch.Tensor): Input sequence of shape (batch_size, sequence_length, num_channels).
            requested_prediction_length (int): Number of future points to predict beyond the input sequence.

        Returns:
            predicted_sequence (torch.Tensor): Predicted sequence of shape (batch_size,
            requested_prediction_length, num_channels).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        total_runs = math.ceil(self.requested_prediction_length / self.model_prediction_length)

        predicted_sequence = past_values.clone()  # Initialize predicted sequence with input sequence

        model_prediction_length = self.model_prediction_length

        this_past = past_values.clone()
        this_past_observed_mask = past_observed_mask.clone() if past_observed_mask is not None else None

        predicted_sequence = None

        prediction_channel_indices = self.model.config.prediction_channel_indices

        for i in range(total_runs):
            # index into the right part of the future
            future_start_idx = i * model_prediction_length
            future_end_idx = (i + 1) * model_prediction_length

            this_future = future_values[:, future_start_idx:future_end_idx] if future_values is not None else None
            this_future_observed_mask = (
                future_observed_mask[:, future_start_idx:future_end_idx] if future_observed_mask is not None else None
            )

            # predict and concatenate results
            next_point = self.model(
                past_values=this_past,
                future_values=this_future,
                past_observed_mask=this_past_observed_mask,
                future_observed_mask=this_future_observed_mask,
                freq_token=freq_token,
                static_categorical_values=static_categorical_values,
            )

            next_point = next_point["prediction_outputs"]
            predicted_sequence = (
                torch.cat((predicted_sequence, next_point), dim=1) if predicted_sequence is not None else next_point
            )

            # create the new part of the past, using the current future and new predictions
            if this_future is not None and prediction_channel_indices is not None:
                new_past = this_future.clone()
                new_past[:, :, self.model.config.prediction_channel_indices] = next_point
            else:
                new_past = next_point

            # update the past observed mask by copying the future
            if this_future_observed_mask is not None and prediction_channel_indices is not None:
                new_past_observed_mask = this_future_observed_mask.clone()
                new_past_observed_mask[:, :, self.model.config.prediction_channel_indices] = True
            else:
                new_past_observed_mask = torch.ones_like(this_future_observed_mask, dtype=torch.bool)

            this_past = torch.cat([this_past[:, model_prediction_length:], new_past], dim=1)
            this_past_observed_mask = (
                torch.cat([this_past_observed_mask[:, model_prediction_length:], new_past_observed_mask], dim=1)
                if this_future_observed_mask is not None
                else None
            )

        output = predicted_sequence

        loss_val = None

        if future_values is not None:
            loss_val = self.loss(output, future_values)
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    output,
                ]
            )

        return RecursivePredictorOutput(
            loss=loss_val,
            prediction_outputs=output,
            backbone_hidden_state=torch.rand(1, 1),
        )
