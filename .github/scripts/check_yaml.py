"""Fail if any workflow or deploy compose file is not valid YAML.

GitHub rejects the ENTIRE workflow when its YAML is invalid, and a push that
breaks one shows up only as a missing run. This runs in CI (`lint` job) so the
breakage fails loudly instead. Requires PyYAML.
"""

import glob
import sys

import yaml

paths = sorted(
    glob.glob(".github/workflows/*.yml")
    + glob.glob(".github/workflows/*.yaml")
    + glob.glob("deploy/**/*.yml", recursive=True)
    + glob.glob("deploy/**/*.yaml", recursive=True)
)
if not paths:
    sys.exit("no YAML files found; run from the repository root")

failed = False
for p in paths:
    try:
        with open(p) as f:
            doc = yaml.safe_load(f)
        if doc is None:
            raise ValueError("empty document")
        if p.startswith(".github/workflows/") and "jobs" not in doc:
            raise ValueError("workflow has no `jobs` key")
        print(f"ok   {p}")
    except Exception as e:  # noqa: BLE001
        failed = True
        print(f"FAIL {p}: {e}")

sys.exit(1 if failed else 0)
