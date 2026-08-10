import unittest

from edgeforge.cli import build_parser


class CLITests(unittest.TestCase):
    def test_kernel_register_uses_only_explicit_dtypes(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "kernel-register",
                "--operator",
                "matmul",
                "--backend",
                "triton",
                "--version",
                "1",
                "--dtype",
                "fp16",
            ]
        )
        self.assertEqual(args.dtype, ["fp16"])

    def test_kernel_register_can_apply_fp32_default_after_parsing(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "kernel-register",
                "--operator",
                "softmax",
                "--backend",
                "python-reference",
                "--version",
                "1",
            ]
        )
        self.assertEqual(args.dtype or ["fp32"], ["fp32"])


if __name__ == "__main__":
    unittest.main()
