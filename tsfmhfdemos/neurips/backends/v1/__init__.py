# Copyright contributors to the TSFM project
#

import toml
import os

_config_file = os.path.join(os.path.dirname(__file__), "config.toml")
GLOBAL_CONFIG = toml.load(open(_config_file))
