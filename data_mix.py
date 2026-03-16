"""
데이터 비중 조절 스크립트 (Data Mixing)

사용법:
    python data_mix.py --input data/raw.jsonl --output data/train.jsonl --weight 3
"""

import argparse
import json
import random


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(args):
    data = load_jsonl(args.input)
    print(f"[로드] {len(data)}개 대화 쌍")

    # 비중 강조: 데이터 반복
    weighted = data * args.weight
    random.shuffle(weighted)

    # 빈 응답 제거
    before = len(weighted)
    weighted = [d for d in weighted if len(d.get("human","")) > 2 and len(d.get("assistant","")) > 2]
    print(f"[정제] 빈 응답 {before - len(weighted)}개 제거")

    save_jsonl(weighted, args.output)
    print(f"[완료] {len(weighted)}개 → {args.output} 저장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight", type=int, default=1, help="반복 횟수 (비중 강조)")
    args = parser.parse_args()
    main(args)
