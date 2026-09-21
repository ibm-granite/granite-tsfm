"""Checks for benchmark config loading and runner defaults without GIFT-Eval."""

import argparse
import ast
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[2] / "notebooks/hfdemo/gift_eval_ensemble"
runner_path = BENCHMARK / "run_gift_eval.py"
tree = ast.parse(runner_path.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_experiment_config")
namespace = {"json": json, "Path": Path, "__file__": str(runner_path)}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner_path), "exec"), namespace)
load_experiment_config = namespace["load_experiment_config"]


class TestGiftExperimentConfig(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment_config()

    def test_default_config_loads_outside_benchmark_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_experiment_config(), self.config)
            finally:
                os.chdir(previous)

    def test_callable_and_cli_share_config_defaults(self):
        # Isolate definitions so this test needs no benchmark runtime dependencies.
        tree = ast.parse((BENCHMARK / "run_gift_eval.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in (
            "run_evaluation", "parse_args"
        )]
        namespace = {
            "argparse": argparse,
            "RUN_DEFAULTS": self.config["defaults"],
            "CONFIGURATIONS": self.config["configurations"],
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), "run_gift_eval.py", "exec"), namespace)
        signature = inspect.signature(namespace["run_evaluation"])
        defaults = {name: arg.default for name, arg in signature.parameters.items()}
        self.assertEqual(defaults, self.config["defaults"])
        self.assertEqual(vars(namespace["parse_args"]([])), self.config["defaults"])
        overrides = namespace["parse_args"]([
            "--seed", "7", "--datasets", "us_births/M", "--device", "cpu", "--skip_processed",
        ])
        self.assertEqual(overrides.seed, 7)
        self.assertEqual(overrides.datasets, ["us_births/M"])
        self.assertEqual(overrides.device, "cpu")
        self.assertTrue(overrides.skip_processed)

    def test_invalid_experiments_fail_clearly(self):
        mutations = [
            lambda c: c["defaults"].update(model_name_config="unknown"),
            lambda c: c["configurations"][c["defaults"]["model_name_config"]].update(ensemble="unknown"),
            lambda c: c["configurations"][c["defaults"]["model_name_config"]].update(model_names=[]),
            lambda c: c.update(quantile_levels=[0.1, 0.5, 0.9]),
            lambda c: c.update(terms=["invalid"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for mutation in mutations:
                config = json.loads(json.dumps(self.config))
                mutation(config)
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
