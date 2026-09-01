#!/usr/bin/env python3
"""Re-point a derived goal-pose dataset's provenance at its current location.

The build records the absolute path it ran at in goal_pose_provenance.json.
train_droid_molmoact2.sh resolves that against DATASET_ROOT and refuses to train
when they differ, which is what happens after the dataset is moved between
machines. This rewrites ONLY the path labels and the two hashes that cover them.

It cannot change which samples are trained on: every artifact hash (the anchor
manifest, meta/stats.json, meta/info.json, ...) is verified before and after, and
the tool aborts if any of them shifts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(recipe_version: int, config: dict) -> str:
    """Mirror of prepare_dataset.py::_config_dict_fingerprint."""
    encoded = json.dumps(
        {"recipe_version": recipe_version, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_report(root: Path, artifacts: dict[str, str]) -> dict[str, bool]:
    return {
        rel: (root / rel).is_file() and sha256(root / rel) == want
        for rel, want in sorted(artifacts.items())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.dataset_root.expanduser().resolve()
    source = args.source_root.expanduser().resolve()
    prov_path = root / "goal_pose_provenance.json"
    succ_path = root / "_SUCCESS.json"

    prov = json.loads(prov_path.read_text())
    succ = json.loads(succ_path.read_text())
    recipe = prov["recipe_version"]

    # Self-test: the fingerprint method must reproduce the stored value from the
    # stored config. If it does not, this tool does not understand the format and
    # must not write anything.
    stored = prov["config_fingerprint"]
    if config_fingerprint(recipe, prov["config"]) != stored:
        print("ABORT: cannot reproduce the stored config_fingerprint", file=sys.stderr)
        return 2
    print(f"  self-test OK: reproduced stored fingerprint {stored[:16]}...")

    before = artifact_report(root, prov["artifact_sha256"])
    if not all(before.values()):
        for rel, ok in before.items():
            if not ok:
                print(f"ABORT: artifact already bad before any edit: {rel}", file=sys.stderr)
        return 2
    print(f"  {len(before)} artifact hashes verified BEFORE")

    old_out = prov["config"]["output_root"]
    old_src = prov["config"]["source_root"]
    if Path(old_out).expanduser().resolve() == root and Path(old_src).expanduser().resolve() == source:
        print("  nothing to do: provenance already points here")
        return 0

    print(f"  output_root: {old_out}\n            -> {root}")
    print(f"  source_root: {old_src}\n            -> {source}")

    if args.dry_run:
        print("  dry run, nothing written")
        return 0

    shutil.copy2(prov_path, prov_path.with_suffix(".json.prepath-backup"))
    shutil.copy2(succ_path, succ_path.with_suffix(".json.prepath-backup"))
    print("  backups written (*.prepath-backup)")

    prov["config"]["output_root"] = str(root)
    prov["config"]["source_root"] = str(source)
    prov["config_fingerprint"] = config_fingerprint(recipe, prov["config"])
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    succ["config_fingerprint"] = prov["config_fingerprint"]
    succ["provenance_sha256"] = sha256(prov_path)
    succ_path.write_text(json.dumps(succ, indent=2, sort_keys=True) + "\n")

    after = artifact_report(root, prov["artifact_sha256"])
    changed = [rel for rel in before if before[rel] != after[rel]]
    if changed:
        print(f"ABORT: artifact hashes changed: {changed}", file=sys.stderr)
        return 2
    print(f"  {len(after)} artifact hashes still verified AFTER (unchanged)")
    print(f"  new config_fingerprint : {prov['config_fingerprint']}")
    print(f"  new provenance_sha256  : {succ['provenance_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
