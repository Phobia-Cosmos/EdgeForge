import unittest

from edgeforge.target_probe import PROBES, _parse_cpu_model, _parse_mem_total, _user_vulkan_icd_probe


class TargetProbeTests(unittest.TestCase):
    def test_parse_linux_memory_kib(self):
        self.assertEqual(_parse_mem_total("MemTotal:       16384000 kB\n"), 16000)

    def test_parse_cpu_model(self):
        self.assertEqual(_parse_cpu_model("Model name: Test CPU\n"), "Test CPU")

    def test_rknn_probe_separates_runtime_files_from_device_node(self):
        self.assertIn("rknn_runtime", PROBES)
        self.assertIn("rknpu", " ".join(PROBES["rknpu_device"]))
        self.assertIn("rknpu_drm_driver", PROBES)
        self.assertIn("rknpu_platform", PROBES)
        self.assertIn("gpu_userspace", PROBES)

    def test_user_vulkan_icd_probe_is_explicit_and_read_only(self):
        command = _user_vulkan_icd_probe("/tmp/edgeforge/mali.json")
        self.assertEqual(command[:2], ["sh", "-lc"])
        self.assertIn("sha256sum", command[2])
        self.assertIn("mali.json", command[2])


if __name__ == "__main__":
    unittest.main()
