# Copyright contributors to the TSFM project
#

"""Tests for HF config registration"""

import pytest

from tsfm_public.toolkit.hf_util import register_config


def test_register_config():
    with pytest.raises(Exception) as ex:
        register_config("patchtst", "PatchTSTConfig", "UnkownRandomModule")

    with pytest.raises(Exception) as ex:
        register_config("patchtst", "PatchTSTConfig2", "transformers")

