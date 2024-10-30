import logging
import os
from importlib import resources

import yaml

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


LOGGER = logging.getLogger(__file__)


def check_ttm_model_path(model_path):
    if "ibm-granite/granite-timeseries-ttm-r1" in model_path or "ibm/TTM" in model_path:
        return 1
    elif "ibm-granite/granite-timeseries-ttm-r2" in model_path:
        return 2
    elif "ibm/ttm-research-r2" in model_path:
        return 3
    else:
        return 0


class GetTTM(TinyTimeMixerForPrediction):
    @classmethod
    def from_pretrained(cls, model_path, context_length=None, forecast_length=None, **kwargs):
        # Custom behavior before calling the superclass method
        LOGGER.info(f"Loading model from: {model_path}")

        model_path_type = check_ttm_model_path(model_path)
        prediction_filter_length = 0
        ttm_model_revision = None
        if model_path_type != 0:
            if context_length is None or forecast_length is None:
                raise ValueError("Provide `context_length` and `forecast_length` for hugginface model path.")

            # Get right TTM model
            config_dir = resources.files("tsfm_public.resources.model_paths_config")

            with open(os.path.join(config_dir, "ttm.yaml"), "r") as file:
                model_revisions = yaml.safe_load(file)

            if forecast_length <= 96:
                selected_forecast_length = 96
            elif forecast_length <= 192:
                selected_forecast_length = 192
            elif forecast_length <= 336:
                selected_forecast_length = 336
            elif forecast_length <= 720:
                selected_forecast_length = 720
            else:
                raise ValueError("Currently supported maximum forecast_length = 720")

            LOGGER.info("Selected forecast_length =", selected_forecast_length)

            prediction_filter_length = forecast_length

            try:
                model_paths = {
                    1: model_revisions["ibm-granite"]["r1"]["revision"],
                    2: model_revisions["ibm-granite"]["r2"]["revision"],
                    3: model_revisions["research-use"]["r2"]["revision"],
                }
                ttm_model_revision = (
                    model_paths.get(model_path_type, {}).get(context_length, {}).get(selected_forecast_length)
                )

            except KeyError:
                raise ValueError("Model not found, possibly because of wrong context_length.")

        # Load model
        if prediction_filter_length == 0:
            model = super().from_pretrained(model_path, revision=ttm_model_revision, **kwargs)
        else:
            model = super().from_pretrained(
                model_path,
                revision=ttm_model_revision,
                prediction_filter_length=prediction_filter_length,
                **kwargs,
            )
        LOGGER.info("Model loaded successfully!")
        LOGGER.info(
            f"[TTM] context_len = {model.config.context_length}, forecast_len = {model.config.prediction_length}"
        )

        return model


if __name__ == "__main__":
    mp = "ibm-granite/granite-timeseries-ttm-r2"
    cl = 512
    fl = 10
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl, dropout=0.4)

    mp = "ibm-granite/granite-timeseries-ttm-r1"
    cl = 1024
    fl = 56
    model = GetTTM.from_pretrained(model_path=mp, context_length=cl, forecast_length=fl, dropout=0.3)

    mp = "/dccstor/tsfm23/vj_share/models/neurips_ttm/512/720/vj_ttm_512_freq-ver12-r-ef42ed98/models/ttm_model/"
    model = GetTTM.from_pretrained(model_path=mp)
