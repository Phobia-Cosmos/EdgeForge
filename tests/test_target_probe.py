import unittest

from edgeforge.target_probe import PROBES, _parse_cpu_model, _parse_mem_total


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


if __name__ == "__main__":
    unittest.main()
