# Copyright contributors to the TSFM project
#

"""Tests for HF config registration"""

import pytest

from tsfm_public.toolkit.hf_util import register_config


def test_register_config():
    with pytest.raises(Exception) as ex:
        register_config("patchtst", "PatchTSTConfig", "UnkownRandomModule")
    assert "Could not load PatchTSTConfig from" in str(ex)

    with pytest.raises(Exception) as ex:
        register_config("patchtst", "PatchTSTConfig2", "transformers")

    assert "Could not find config for PatchTSTConfig2" in str(ex)
