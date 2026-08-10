import logging
import tempfile
import unittest
from pathlib import Path

from edgeforge.logging_utils import configure_logging


class LoggingTests(unittest.TestCase):
    def test_logs_are_scoped_by_version_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = configure_logging("worker", "0.2.0", directory)
            logging.getLogger("edgeforge.test").warning("persistent event")
            logging.shutdown()
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.parent.name, "v0.2.0")
            content = path.read_text()
            self.assertIn('"version":"0.2.0"', content)
            self.assertIn("persistent event", content)
            self.assertTrue(list(Path(directory, "v0.2.0", "worker").glob("*.jsonl")))


if __name__ == "__main__":
    unittest.main()
