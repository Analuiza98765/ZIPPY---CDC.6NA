import builtins
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VULNERABLE = load_module(
    "zippy_vulnerable_dynamic_tests", ROOT / "Projeto_Zippy_vulneravel.py"
)
FINAL = load_module("zippy_final_dynamic_tests", ROOT / "Projeto_Zippy.py")


class DynamicSecurityChecks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="zippy-security-test-")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_benign_scenario_pose_loads(self):
        Path(VULNERABLE.SCENARIO_FILE).write_text(
            "(5.2, 3.4, 0.0)", encoding="utf-8"
        )
        self.assertEqual(VULNERABLE.load_scenario(), (5.2, 3.4, 0.0))

    def test_malformed_scenario_is_rejected(self):
        Path(VULNERABLE.SCENARIO_FILE).write_text("(", encoding="utf-8")
        self.assertIsNone(VULNERABLE.load_scenario())

    def test_scenario_loader_reaches_eval_without_evaluating_input(self):
        marker = "benign test marker"
        Path(VULNERABLE.SCENARIO_FILE).write_text(marker, encoding="utf-8")
        with patch.object(
            builtins, "eval", side_effect=RuntimeError("blocked by test stub")
        ) as eval_stub:
            self.assertIsNone(VULNERABLE.load_scenario())
        eval_stub.assert_called_once_with(marker)

    def test_save_loader_reaches_pickle_stub_without_deserialization(self):
        sim = VULNERABLE.Sim()
        placeholder = b"benign placeholder; not a pickle"
        Path(VULNERABLE.SAVE_FILE).write_bytes(placeholder)
        safe_data = {
            "x": 1.25,
            "y": 2.5,
            "th": 0.75,
            "dist": 3.0,
            "collisions": 0,
            "trail": [(1.25, 2.5)],
        }
        observed = []

        def pickle_load_stub(file_obj):
            observed.append(file_obj.read())
            return safe_data

        with patch.object(VULNERABLE.pickle, "load", side_effect=pickle_load_stub):
            sim.load_progress()

        self.assertEqual(observed, [placeholder])
        self.assertEqual((sim.robot.x, sim.robot.y, sim.robot.th), (1.25, 2.5, 0.75))

    def test_final_version_has_no_scenario_or_save_loaders(self):
        Path(VULNERABLE.SCENARIO_FILE).write_text(
            "(5.2, 3.4, 0.0)", encoding="utf-8"
        )
        sim = FINAL.Sim()
        expected_x, expected_y = FINAL.px2m(*FINAL.START_PX)

        self.assertAlmostEqual(sim.robot.x, expected_x)
        self.assertAlmostEqual(sim.robot.y, expected_y)
        self.assertFalse(hasattr(FINAL, "load_scenario"))
        self.assertFalse(hasattr(sim, "load_progress"))
        self.assertFalse(hasattr(sim, "save_progress"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
