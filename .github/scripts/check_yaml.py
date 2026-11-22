"""Fail if any workflow or deploy compose file is not valid YAML.

GitHub rejects the ENTIRE workflow when its YAML is invalid, and a push that
breaks one shows up only as a missing run. This runs in CI (`lint` job) so the
breakage fails loudly instead. Requires PyYAML.
"""

import glob
import sys

import yaml

WORKFLOWS = ".github/workflows/"


def is_workflow(path):
    """True for a file under .github/workflows/, whatever separator glob used.

    On Windows glob returns `.github/workflows\\ci.yml`, so a plain
    `startswith(".github/workflows/")` never matched there and the `jobs`
    check below was silently skipped.
    """
    return path.replace("\\", "/").startswith(WORKFLOWS)


paths = sorted(
    glob.glob(".github/workflows/*.yml")
    + glob.glob(".github/workflows/*.yaml")
    + glob.glob("deploy/**/*.yml", recursive=True)
    + glob.glob("deploy/**/*.yaml", recursive=True)
)
if not paths:
    sys.exit("no YAML files found; run from the repository root")
if not any(is_workflow(p) for p in paths):
    sys.exit("no workflow recognised under .github/workflows; the `jobs` check would not run")

failed = False
for p in paths:
    try:
        # The files are UTF-8; do not depend on the platform's default encoding.
        with open(p, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if doc is None:
            raise ValueError("empty document")
        if is_workflow(p) and "jobs" not in doc:
            raise ValueError("workflow has no `jobs` key")
        print(f"ok   {p}")
    except Exception as e:  # noqa: BLE001
        failed = True
        print(f"FAIL {p}: {e}")

sys.exit(1 if failed else 0)
