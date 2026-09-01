import unittest

try:
    import torch
except ImportError:  # Core EdgeForge remains usable without the optional torch dependency.
    torch = None

from edgeforge.eeg_models import (
    EEGModelConfig,
    available_eeg_decoders,
    build_eeg_decoder,
    canonical_decoder_name,
)


class EEGRegistryTests(unittest.TestCase):
    def test_builtin_names_and_aliases_are_stable(self):
        names = available_eeg_decoders()
        for name in (
            "eegnet",
            "tcn",
            "transformer",
            "shallowconvnet",
            "deepconvnet",
            "fbcnet",
            "conformer",
            "tsception",
            "atcnet",
            "cnn_lstm",
            "eeg_graph",
            "brainuicl",
            "lop_mlp",
        ):
            self.assertIn(name, names)
        self.assertEqual(canonical_decoder_name("eeg_net"), "eegnet")
        self.assertEqual(canonical_decoder_name("brainuicl/lop_mlp"), "lop_mlp")
        self.assertEqual(canonical_decoder_name("dgcnn"), "eeg_graph")

    def test_mapping_config_preserves_custom_options(self):
        config = EEGModelConfig.from_mapping({"name": "lop_mlp", "hidden_dims": [8, 12]})
        self.assertEqual(config.options["hidden_dims"], [8, 12])
        self.assertEqual(config.to_dict()["dilations"], [1, 2, 4])

    def test_dual_branch_in_channels_infers_eeg_remainder(self):
        config = EEGModelConfig(name="brainuicl", in_channels=8, eog_channels=2)
        self.assertEqual(config.eeg_channels, 6)


@unittest.skipUnless(torch is not None, "PyTorch is required for numerical decoder tests")
class EEGDecoderNumericalTests(unittest.TestCase):
    def test_all_decoders_return_bundle_for_epoch_and_sequence_inputs(self):
        x = torch.randn(2, 8, 128)
        sequence = torch.randn(2, 3, 8, 128)
        names = available_eeg_decoders()
        for name in names:
            options = {"d_model": 32} if name == "brainuicl" else {}
            config = EEGModelConfig(
                name=name,
                in_channels=8,
                eeg_channels=8,
                input_length=128,
                num_classes=5,
                feature_dim=16,
                width=16,
                heads=4,
                layers=1,
                depth=2,
                options=options,
            )
            model = build_eeg_decoder(config).eval()
            with torch.no_grad():
                epoch_bundle = model.forward_bundle(x)
                sequence_bundle = model.forward_bundle(sequence)
            self.assertEqual(tuple(epoch_bundle.logits.shape), (2, 5), name)
            self.assertEqual(tuple(sequence_bundle.logits.shape), (2, 3, 5), name)
            self.assertTrue(epoch_bundle.representations, name)
            self.assertEqual(model(x).shape, (2, 5), name)

    def test_brainuicl_dual_branch_and_attention_metadata(self):
        config = EEGModelConfig(
            name="brainuicl",
            in_channels=8,
            eeg_channels=8,
            eog_channels=2,
            input_length=128,
            num_classes=3,
            feature_dim=16,
            heads=4,
            layers=1,
            options={"d_model": 32},
        )
        model = build_eeg_decoder(config).eval()
        eeg = torch.randn(2, 3, 8, 128)
        eog = torch.randn(2, 3, 2, 128)
        with torch.no_grad():
            bundle = model.forward_bundle((eeg, eog))
        self.assertEqual(tuple(bundle.logits.shape), (2, 3, 3))
        self.assertIn("transformer.0", bundle.attention)
        self.assertEqual(bundle.metadata["attention_normalization_axis"], 1)

    def test_brainuicl_single_branch_accepts_combined_or_separate_inputs(self):
        config = EEGModelConfig(
            name="brainuicl",
            in_channels=10,
            eeg_channels=8,
            eog_channels=0,
            input_length=64,
            num_classes=3,
            feature_dim=8,
            heads=2,
            layers=1,
            options={"d_model": 16},
        )
        model = build_eeg_decoder(config).eval()
        eeg = torch.randn(2, 8, 64)
        eog = torch.randn(2, 2, 64)
        with torch.no_grad():
            combined = model.forward_bundle(torch.cat((eeg, eog), dim=1))
            separate = model.forward_bundle((eeg, eog))
        self.assertEqual(tuple(combined.logits.shape), (2, 3))
        self.assertEqual(tuple(separate.logits.shape), (2, 3))

    def test_short_windows_and_even_lengths_keep_all_decoders_shape_safe(self):
        """Registry models should remain usable for tiny calibration windows."""

        for name in available_eeg_decoders():
            config = EEGModelConfig(
                name=name,
                in_channels=4,
                eeg_channels=4,
                input_length=4,
                num_classes=3,
                feature_dim=8,
                width=8,
                depth=2,
                layers=1,
                heads=2,
                patch_size=4,
                dropout=0.0,
                options={"d_model": 8},
            )
            model = build_eeg_decoder(config).eval()
            with torch.no_grad():
                output = model(torch.randn(2, 4, 1))
            self.assertEqual(tuple(output.shape), (2, 3), name)


if __name__ == "__main__":
    unittest.main()
