# Copyright contributors to the TSFM project
#

"""Tests the base processor class"""

import tempfile

from tsfm_public.toolkit.processor import BaseProcessor


def test_processor_saves_with_custom_name():
    class MyProcessor(BaseProcessor):
        PROCESSOR_NAME = "my_processor_config.json"

    p = MyProcessor(my_param="10")

    with tempfile.TemporaryDirectory() as d:
        p.save_pretrained(d)
        p_new = MyProcessor.from_pretrained(d)

        assert p.my_param == p_new.my_param
