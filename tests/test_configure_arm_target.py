import unittest
from pathlib import Path


class ConfigureArmTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        path = Path(__file__).parents[1] / "scripts" / "configure-arm-target.py"
        spec = importlib.util.spec_from_file_location("configure_arm_target", path)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)

    def test_inventory_json_parser_rejects_non_object(self):
        self.assertIsNone(self.module._parse_json("[]"))
        self.assertEqual(self.module._parse_json('{"architecture":"aarch64"}')['architecture'], "aarch64")

    def test_prepare_script_keeps_rknn_blocked_and_does_not_write_token(self):
        script = self.module._remote_prepare("orangepi", self.module.TARGETS["orangepi"])
        self.assertIn("python-reference", script)
        self.assertIn("blocked-until-device-and-correctness", script)
        self.assertIn("credentials_written", script)
        self.assertIn("EDGEFORGE_TOKEN=replace-with-runtime-secret", script)

    def test_target_profiles_are_explicit(self):
        self.assertEqual(self.module.TARGETS["orangepi"]["architecture"], "aarch64")
        self.assertEqual(self.module.TARGETS["p550"]["architecture"], "riscv64")


if __name__ == "__main__":
    unittest.main()
