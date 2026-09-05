"""Exercise corpus extraction with small, checksum-pinned local ZIP fixtures."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "t834-conformance",
        "ITU-T_T.834(2014-10)_ConformanceSuite.zip",
        "JXR_ConformanceSuite_2014/vector.bin",
        "suite-2014/vector.bin",
    ),
    (
        "t835-oracle",
        "T-REC-T.835-201201-S.zip",
        "Software/Makefile",
        "t835-201201/Software/Makefile",
    ),
)


class CorpusBuildTests(unittest.TestCase):
    def run_fixture(self, case, *, with_unzip=False, corrupt=False):
        tool, archive_name, member, extracted = case
        with tempfile.TemporaryDirectory(prefix="jxr-corpus-test-") as directory:
            workspace = Path(directory)
            script = workspace / "tools" / tool / "build.sh"
            script.parent.mkdir(parents=True)
            suite = workspace / "target" / tool
            archive = suite / "downloads" / archive_name
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(member, b"fixture bytes\n")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            source = (ROOT / "tools" / tool / "build.sh").read_text()
            source, replacements = re.subn(
                r"archive_sha256='[0-9a-f]{64}'",
                f"archive_sha256='{digest}'",
                source,
            )
            self.assertEqual(replacements, 1)
            script.write_text(source)
            if corrupt:
                archive.write_bytes(archive.read_bytes() + b"corrupted")

            # Limit PATH to reproduce the CUDA runner's missing unzip utility.
            binaries = workspace / "bin"
            binaries.mkdir()
            for name in ("dirname", "mkdir", "mktemp", "awk", "mv", "shasum"):
                executable = shutil.which(name)
                self.assertIsNotNone(executable, name)
                (binaries / name).symlink_to(executable)
            (binaries / "python3").symlink_to(sys.executable)
            if with_unzip:
                executable = shutil.which("unzip")
                self.assertIsNotNone(executable, "unzip")
                (binaries / "unzip").symlink_to(executable)
            # Oracle compilation is a separate boundary; verify it gets real sources.
            make = binaries / "make"
            make.write_text('#!/bin/sh\n[ "$1" = -C ] && [ -f "$2/Makefile" ]\n')
            make.chmod(0o755)
            result = subprocess.run(
                ["/bin/sh", str(script)],
                env={**os.environ, "PATH": str(binaries)},
                capture_output=True,
                text=True,
                check=False,
            )
            if corrupt:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("archive checksum mismatch", result.stderr)
                self.assertFalse((suite / extracted).exists())
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((suite / extracted).read_bytes(), b"fixture bytes\n")

    def test_extracts_without_unzip(self):
        for case in CASES:
            with self.subTest(tool=case[0]):
                self.run_fixture(case)

    def test_existing_unzip_route(self):
        for case in CASES:
            with self.subTest(tool=case[0]):
                self.run_fixture(case, with_unzip=True)

    def test_rejects_corrupt_archive_before_extraction(self):
        for case in CASES:
            with self.subTest(tool=case[0]):
                self.run_fixture(case, corrupt=True)


if __name__ == "__main__":
    unittest.main()
