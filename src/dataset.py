# dataset.py
import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RefusalPair:
    pair_id: str
    knowledge_question: str
    safety_question: str
    category: str
    judge_decision: str = "keep"
    # control prompts (normal answers) — populated when control_dual schema detected
    k_control_question: Optional[str] = None
    s_control_question: Optional[str] = None


def load_valid_pairs(path: str, max_pairs: Optional[int] = None) -> List[RefusalPair]:
    """
    Load valid pairs from either schema:
      - legacy:       knowledge_question / safety_question + judge.decision filter
      - control_dual: k_prompt / s_prompt + pair_validation.final_pass filter
                      includes k_control_prompt / s_control_prompt as normal answers
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Detect schema
            if "k_prompt" in item:
                # control_dual schema
                if not item.get("pair_validation", {}).get("final_pass", True):
                    continue
                pairs.append(RefusalPair(
                    pair_id=item["pair_id"],
                    knowledge_question=item["k_prompt"],
                    safety_question=item["s_prompt"],
                    category=item.get("category", "unknown"),
                    judge_decision="keep",
                    k_control_question=item.get("k_control_prompt"),
                    s_control_question=item.get("s_control_prompt"),
                ))
            else:
                # legacy schema
                decision = item.get("judge", {}).get("decision", "")
                if decision not in {"keep", "revise"}:
                    continue
                pairs.append(RefusalPair(
                    pair_id=item["pair_id"],
                    knowledge_question=item["knowledge_question"],
                    safety_question=item["safety_question"],
                    category=item.get("category", "unknown"),
                    judge_decision=decision,
                ))

    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def has_control_prompts(pairs: List[RefusalPair]) -> bool:
    return all(
        p.k_control_question is not None and p.s_control_question is not None
        for p in pairs
    )
