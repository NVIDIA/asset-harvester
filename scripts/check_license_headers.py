#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that all Python source files begin with the required SPDX / Apache-2.0
license header.

Usage:
    python scripts/check_license_headers.py          # check all .py files
    python scripts/check_license_headers.py --fix     # prepend header to files that lack it
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

HEADER_LINES = EXPECTED_HEADER.strip().splitlines()

EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".eggs"}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def collect_py_files() -> list[Path]:
    return sorted(p for p in REPO_ROOT.rglob("*.py") if not _is_excluded(p.relative_to(REPO_ROOT)))


def _strip_shebang(lines: list[str]) -> list[str]:
    """Skip a leading shebang so the header can appear right after it."""
    if lines and lines[0].startswith("#!"):
        return lines[1:]
    return lines


def check_header(path: Path) -> bool:
    """Return True if *path* starts with the expected header."""
    text = path.read_text(encoding="utf-8", errors="replace")
    file_lines = _strip_shebang(text.splitlines())
    if len(file_lines) < len(HEADER_LINES):
        return False
    return file_lines[: len(HEADER_LINES)] == HEADER_LINES


def fix_header(path: Path) -> None:
    """Prepend the license header to *path* (preserving any shebang)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    shebang = ""
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        lines = lines[1:]

    new_text = shebang + EXPECTED_HEADER + "\n" + "".join(lines)
    path.write_text(new_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Prepend the license header to files that are missing it.",
    )
    args = parser.parse_args()

    py_files = collect_py_files()
    failures: list[Path] = []

    for path in py_files:
        if not check_header(path):
            failures.append(path)

    if not failures:
        print(f"All {len(py_files)} files have the correct license header.")
        return 0

    if args.fix:
        for path in failures:
            fix_header(path)
            rel = path.relative_to(REPO_ROOT)
            print(f"  fixed: {rel}")
        print(f"\nPrepended license header to {len(failures)} file(s).")
        return 0

    print(f"Found {len(failures)} file(s) missing the license header:\n")
    for path in failures:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print("\nRun with --fix to auto-prepend the header.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
