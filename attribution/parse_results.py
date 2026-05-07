"""
Load amao's OLMo2-1B results for both conditions and return matched
(prompt, gold) pairs for examples where the model is correct in both.
"""

import json
from pathlib import Path
from dataclasses import dataclass

from config import IN_CTX_RESULTS, RECALL_RESULTS, ATTR_RESULTS_DIR


@dataclass
class Sample:
    example_id: int
    prompt: str
    gold: str
    corrupted: str


def load_all(path: Path) -> dict[int, dict]:
    """Return {example_id: raw_result_dict} for every example in the file."""
    with open(path) as f:
        data = json.load(f)
    return {r["example_id"]: r for r in data["results"]}


def get_matched_samples() -> tuple[list[Sample], list[Sample]]:
    """
    Return (in_cxt_samples, recall_samples) for the intersection of
    correctly predicted examples across both conditions, sorted by example_id.

    The `corrupted` field is the gold of the *other* condition at the same
    example_id: for in-cxt, the corrupted token is the parametric/recall
    answer; for recall, it is the in-context distractor.
    """
    in_cxt_all = load_all(IN_CTX_RESULTS)
    recall_all = load_all(RECALL_RESULTS)

    shared_correct_ids = sorted(
        i for i in (in_cxt_all.keys() & recall_all.keys())
        if in_cxt_all[i]["correct"] and recall_all[i]["correct"]
        and i != 65
    )

    in_cxt_samples = [
        Sample(
            example_id=i,
            prompt=in_cxt_all[i]["icl_prompt"],
            gold=in_cxt_all[i]["gold"],
            corrupted=recall_all[i]["gold"],
        )
        for i in shared_correct_ids
    ]
    recall_samples = [
        Sample(
            example_id=i,
            prompt=recall_all[i]["icl_prompt"],
            gold=recall_all[i]["gold"],
            corrupted=in_cxt_all[i]["gold"],
        )
        for i in shared_correct_ids
    ]

    return in_cxt_samples, recall_samples


if __name__ == "__main__":
    in_cxt_samples, recall_samples = get_matched_samples()

    print(f"Matched examples: {len(in_cxt_samples)}")
    print("\nFirst in-context sample:")
    print(f"  example_id : {in_cxt_samples[0].example_id}")
    print(f"  gold       : {in_cxt_samples[0].gold}")
    print(f"  corrupted  : {in_cxt_samples[0].corrupted}")
    print(f"  prompt     :\n{in_cxt_samples[0].prompt[:300]}...")
    print("\nCorresponding recall sample:")
    print(f"  example_id : {recall_samples[0].example_id}")
    print(f"  gold       : {recall_samples[0].gold}")
    print(f"  corrupted  : {recall_samples[0].corrupted}")
    print(f"  prompt     :\n{recall_samples[0].prompt[:300]}...")

    out = ATTR_RESULTS_DIR / "matched_example_ids.json"
    with open(out, "w") as f:
        json.dump([s.example_id for s in in_cxt_samples], f, indent=2)
    print(f"\nSaved matched example IDs to {out}")
