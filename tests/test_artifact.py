import tempfile
import unittest
from pathlib import Path

from edgeforge.artifact import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_content_is_immutable_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = store.put(b"same-content", kind="manifest", media_type="application/json", name="first.json")
            second = store.put(b"same-content", kind="manifest", media_type="application/json", name="second.json")
            self.assertEqual(first["digest"], second["digest"])
            self.assertEqual(store.read(first["digest"]), b"same-content")
            blobs = list(Path(directory).glob("sha256/*/*/blob"))
            self.assertEqual(len(blobs), 1)

    def test_invalid_digest_and_oversized_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, max_bytes=4)
            with self.assertRaises(ValueError):
                store.put(b"12345", kind="test", media_type="text/plain", name="large")
            with self.assertRaises(ValueError):
                store.read("not-a-digest")


if __name__ == "__main__":
    unittest.main()

