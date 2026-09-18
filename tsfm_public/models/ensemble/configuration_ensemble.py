# Copyright contributors to the TSFM project
#
"""Hugging Face configuration for a probabilistic ensemble recipe."""

from copy import deepcopy
from math import isclose, isfinite
from numbers import Real

from transformers import PretrainedConfig


DEFAULT_MEMBERS = [
    {"forecaster_type": "patchtst", "model_checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r1"},
    {"forecaster_type": "patchtst", "model_checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r2"},
    {
        "forecaster_type": "flowstate",
        "model_checkpoint": "ibm-granite/granite-timeseries-flowstate-r1",
        "model_revision": "r1.1",
    },
    {"forecaster_type": "ttm", "model_checkpoint": "ibm-granite/granite-timeseries-ttm-r3"},
]


class ProbabilisticEnsembleConfig(PretrainedConfig):
    """A recipe of member checkpoints, quantiles, and aggregation settings.

    Use inherited ``from_pretrained`` and ``save_pretrained`` methods with a local
    directory or Hugging Face repository. Device and forecast lengths are runtime
    choices. TTM revisions are selected dynamically, never stored in this recipe.

    Args:
        members: Ordered member definitions with ``forecaster_type`` (``patchtst``,
            ``flowstate``, or ``ttm``) and ``model_checkpoint``. Defaults to Granite
            PatchTST-r1, PatchTST-r2, FlowState-r1.1, and TTM-r3. TTM must not specify
            ``model_revision``; its revision is selected by ``get_model()``.
        quantile_levels: Increasing, unique levels within (0, 1), including 0.5.
            Defaults to 0.1 through 0.9 in increments of 0.1.
        aggregation_method: Supported values are ``linear_pool`` (default), which
            pools member quantile forecasts, and ``iqr_weighted``, which gives
            greater weight to members with narrower interquantile ranges.
        iqr_weighted_options: Options used only by ``iqr_weighted``. ``temperature``
            defaults to 0.5 and must be finite and positive; lower values concentrate
            weights, while higher values move them toward uniform weighting.
            ``max_weight`` defaults to 0.4 and caps each member's aggregation weight;
            it must be between ``1 / len(members)`` and 1, or None to disable capping.
            Narrower intervals do not necessarily indicate greater accuracy.
        weights: Optional non-negative member weights in member order, summing to 1.
            Defaults to None (equal influence for linear pooling; weights derived
            from IQR for IQR weighting). For ``iqr_weighted``, supplied weights act
            as multiplicative priors on the IQR-derived weights.

    Example:
        >>> config = ProbabilisticEnsembleConfig.from_pretrained("./ensemble_recipe")
        >>> config.aggregation_method = "iqr_weighted"
        >>> config.iqr_weighted_options["temperature"] = 0.8
        >>> config.validate()
        >>> config.save_pretrained("./my_ensemble_recipe")
    """

    model_type = "granite_probabilistic_ensemble"

    def __init__(
        self,
        members=None,
        quantile_levels=None,
        aggregation_method="linear_pool",
        iqr_weighted_options=None,
        weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.members = deepcopy(DEFAULT_MEMBERS if members is None else members)
        self.quantile_levels = list(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if quantile_levels is None else quantile_levels
        )
        self.aggregation_method = aggregation_method
        self.iqr_weighted_options = {"temperature": 0.5, "max_weight": 0.4}
        if iqr_weighted_options is not None:
            self.iqr_weighted_options.update(iqr_weighted_options)
        self.weights = list(weights) if weights is not None else None
        self.validate()

    def validate(self):
        """Validate the recipe before any member weights are loaded."""
        if self.aggregation_method not in ("linear_pool", "iqr_weighted"):
            raise ValueError("aggregation_method must be 'linear_pool' or 'iqr_weighted'.")
        if not isinstance(self.members, list) or not self.members:
            raise ValueError("members must be a non-empty list of member definitions.")
        member_options = {
            "patchtst": {"forecaster_type", "model_checkpoint"},
            "flowstate": {"forecaster_type", "model_checkpoint", "model_revision", "scale_factor", "batch_first"},
            "ttm": {"forecaster_type", "model_checkpoint"},
        }
        for member in self.members:
            if not isinstance(member, dict) or member.get("forecaster_type") not in member_options:
                raise ValueError("Each member must have forecaster_type 'patchtst', 'flowstate', or 'ttm'.")
            checkpoint = member.get("model_checkpoint")
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError("Each member must provide a non-empty model_checkpoint.")
            if member["forecaster_type"] == "ttm" and "model_revision" in member:
                raise ValueError("TTM model_revision must remain dynamically selected by get_model().")
            unknown = set(member) - member_options[member["forecaster_type"]]
            if unknown:
                raise ValueError(f"Unsupported member options: {sorted(unknown)}")
        levels = self.quantile_levels
        if (
            not levels
            or any(not isinstance(q, Real) or not isfinite(q) or not 0 < q < 1 for q in levels)
            or levels != sorted(set(levels))
            or 0.5 not in levels
        ):
            raise ValueError("quantile_levels must be increasing, unique, within (0, 1), and include 0.5.")
        options = self.iqr_weighted_options
        if not isinstance(options, dict) or set(options) != {"temperature", "max_weight"}:
            raise ValueError("iqr_weighted_options supports only temperature and max_weight.")
        temperature = options["temperature"]
        if not isinstance(temperature, Real) or not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero.")
        cap = options["max_weight"]
        if self.aggregation_method == "iqr_weighted" and cap is not None:
            if not isinstance(cap, Real) or not isfinite(cap) or not 1 / len(self.members) <= cap <= 1:
                raise ValueError("max_weight must be between 1 / n_members and 1, or None.")
        if self.weights is not None:
            if (
                len(self.weights) != len(self.members)
                or any(not isinstance(w, Real) or not isfinite(w) or w < 0 for w in self.weights)
                or not isclose(sum(self.weights), 1.0)
            ):
                raise ValueError("weights must contain one finite non-negative value per member and sum to 1.")
