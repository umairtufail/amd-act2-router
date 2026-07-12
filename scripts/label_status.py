"""Print labeling progress toward a target total."""

import argparse
import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "tasks_raw.jsonl"
LABELED = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, default=0,
        help="desired labeled total; 0 (default) means every raw task",
    )
    args = parser.parse_args()

    raw_n = sum(1 for _ in open(RAW, encoding="utf-8") if _.strip())
    target = args.target or raw_n
    if not LABELED.exists():
        print(f"Labeled: 0 / {raw_n} (target {target})")
        return
    rows = [json.loads(l) for l in open(LABELED, encoding="utf-8") if l.strip()]
    c = Counter(r["tier_label"] for r in rows)
    done_ids = {r["id"] for r in rows}
    missing = raw_n - len(done_ids)
    print(f"Labeled: {len(rows)} / {raw_n} (target {target}, need {max(target - len(rows), 0)} more)")
    print(f"Tiers: {dict(c)}")
    print(f"Missing IDs: {missing}")


if __name__ == "__main__":
    main()
