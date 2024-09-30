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
            Prediction output from the forecast head.
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
        model (PreTrainedModel): Model to load for recursive forecasts
        requested_forecast_length (int): Total forecast length
        model_forecast_length (int): forecast length of the model
        loss (str): loss to report
    """

    def __init__(
        self, model: PreTrainedModel, requested_forecast_length: int, model_forecast_length: int, loss: str, **kwargs
    ):
        self.model = model
        self.requested_forecast_length = requested_forecast_length
        self.model_forecast_length = model_forecast_length
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
        self.requested_forecast_length = config.requested_forecast_length
        self.model_forecast_length = config.model_forecast_length
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
        return_dict: Optional[bool] = None,
        return_loss: bool = True,
        freq_token: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RecursivePredictorOutput:
        """
        Predict future points given an input sequence.

        Args:
            past_values (torch.Tensor): Input sequence of shape (batch_size, sequence_length, num_channels).
            self.requested_forecast_length (int): Number of future points to predict beyond the input sequence.

        Returns:
            predicted_sequence (torch.Tensor): Predicted sequence of shape (batch_size, self.requested_forecast_length, num_channels).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        total_runs = math.ceil(self.requested_forecast_length / self.model_forecast_length)
        # device = past_values.device
        device = past_values.device  # Get device of input_sequence

        # self.model.to(device)  # Move model to the same device as input_sequence
        past_values = past_values.to(device)

        # device = next(self.model.parameters()).device
        # with torch.no_grad():
        sequence_length = past_values.size(1)
        predicted_sequence = past_values.clone()  # Initialize predicted sequence with input sequence

        for i in range(total_runs):
            # Predict the next time step
            next_point = self.model(predicted_sequence[:, -sequence_length:], freq_token=freq_token)
            next_point = next_point["prediction_outputs"]
            predicted_sequence = torch.cat((predicted_sequence, next_point), dim=1)

        output = predicted_sequence[:, -self.requested_forecast_length :]  # Return only the predicted future points

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
