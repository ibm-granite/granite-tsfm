# Copyright contributors to the TSFM project
#

"""Tests conformal processor capabilities"""

import tempfile
from pathlib import Path

from tsfm_public.toolkit.conformal import PostHocProbabilisticProcessor


def test_conformal_save_pretrained():
    # initial test to check that we save the ProbabbilisticProcessor as intended
    p = PostHocProbabilisticProcessor()

    with tempfile.TemporaryDirectory() as d:
        p.save_pretrained(d)
        p_new = PostHocProbabilisticProcessor.from_pretrained(d)
        assert Path(d).joinpath(PostHocProbabilisticProcessor.PROCESSOR_NAME).exists()

        # to do: add checks that p and p_new are equivalent
