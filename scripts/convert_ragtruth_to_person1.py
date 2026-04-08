from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.conversion import convert_ragtruth_to_person1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAGTruth files to Person 1 pipeline JSONL schema"
    )
    parser.add_argument(
        "--response-jsonl",
        type=Path,
        default=Path("RAGTruth-main/dataset/response.jsonl"),
        help="Path to RAGTruth response.jsonl",
    )
    parser.add_argument(
        "--source-info-jsonl",
        type=Path,
        default=Path("RAGTruth-main/dataset/source_info.jsonl"),
        help="Path to RAGTruth source_info.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/ragtruth/raw.jsonl"),
        help="Output file path in Person 1 schema",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke tests (0 = all rows)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_ragtruth_to_person1(
        response_jsonl=args.response_jsonl,
        source_info_jsonl=args.source_info_jsonl,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
