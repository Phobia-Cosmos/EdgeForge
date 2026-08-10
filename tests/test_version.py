import unittest
from pathlib import Path

from edgeforge import __version__


class VersionTests(unittest.TestCase):
    def test_package_and_release_metadata_match(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text()
        self.assertIn(f'version = "{__version__}"', pyproject)
        self.assertTrue((root / "releases" / f"v{__version__}.md").is_file())
        self.assertIn(f"## {__version__} -", (root / "CHANGELOG.md").read_text())


if __name__ == "__main__":
    unittest.main()

