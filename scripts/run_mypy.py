#!/usr/bin/env python
"""Run mypy over the package and fail on any error, grouping output for readability.

Usage
-----
python scripts/run_mypy.py [--groupby {file|errorcode|message}]
"""

import argparse
import io
import subprocess
import sys

import polars as pl

PACKAGE = "time_series_transformers"


def mypy_to_polars(mypy_result: str) -> pl.DataFrame:
    """Reformat mypy JSON-lines output into a DataFrame.

    Adapted from: https://gist.github.com/michaelosthege/24d0703e5f37850c9e5679f69598930a
    """
    if not mypy_result.strip():
        return pl.DataFrame(
            schema={
                "file": pl.Utf8,
                "line": pl.Int64,
                "code": pl.Utf8,
                "severity": pl.Utf8,
                "message": pl.Utf8,
            }
        )
    return pl.read_ndjson(io.StringIO(mypy_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run mypy type checks on the {PACKAGE} codebase.")
    parser.add_argument(
        "--groupby",
        default="file",
        help="How to group the output. One of {file|errorcode|message}.",
    )
    args, _ = parser.parse_known_args()

    cp = subprocess.run(
        [
            "mypy",
            "--output",
            "json",
            "--disable-error-code",
            "annotation-unchecked",
            PACKAGE,
        ],
        capture_output=True,
        check=False,
    )

    df = mypy_to_polars(cp.stdout.decode("utf-8"))

    if df.is_empty():
        print(f"All files in {PACKAGE} pass the mypy type checks.")
        sys.exit(0)

    for (section,), sdf in df.group_by(args.groupby, maintain_order=True):
        print(f"\n\n[{section}]")
        for row in sdf.iter_rows(named=True):
            print(
                f"{row['file']}:{row['line']}: {row['code']} [{row['severity']}]: {row['message']}"
            )
    print()
    sys.exit(1)
