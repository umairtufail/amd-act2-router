"""Generate deterministic, family-separated candidates for hard-case mining.

This module performs no model calls.  Entire prompt/template families are
assigned to either training or stress holdout *before* tier0 is queried, which
prevents outcome-aware splitting and paraphrase-family leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.integrity import (
    assert_disjoint_mining_partitions,
    assert_no_prompt_overlap,
    dataset_hash,
    prompt_family_key,
    prompt_group_key,
    stable_json_hash,
)
from data.schema import TaskExample

GENERATOR_VERSION = "hard-candidates-v1"
DATA_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = DATA_DIR / "mining"
RAW_TASKS_PATH = DATA_DIR / "tasks_raw.jsonl"
HOLDOUT_TASKS_PATH = DATA_DIR / "holdout_tasks.json"


@dataclass(frozen=True)
class CandidateFamily:
    category: str
    family_id: str
    # difficulty, prompt, ground truth
    examples: tuple[tuple[str, str, str], ...]


def _ner_prompt(sentence: str) -> str:
    return (
        "Extract all named entities from the sentence below. Respond with ONLY "
        'a JSON object with keys "persons", "organizations", and "locations", '
        f"each a list of strings.\n\nSentence: {sentence}"
    )


def _ner_truth(persons: list[str], organizations: list[str], locations: list[str]) -> str:
    return json.dumps(
        {
            "persons": persons,
            "organizations": organizations,
            "locations": locations,
        },
        sort_keys=True,
    )


def _ner_families() -> list[CandidateFamily]:
    specifications: list[tuple[str, list[tuple[str, str, list[str], list[str], list[str]]]]] = [
        (
            "ner_surface_type_collision",
            [
                ("adversarial", "Jordan Lake of Lake Research briefed officials in Reading, England.", ["Jordan Lake"], ["Lake Research"], ["Reading, England"]),
                ("adversarial", "Brooklyn Fields from Fields Group moved from Victoria, Canada to Brooklyn, New York.", ["Brooklyn Fields"], ["Fields Group"], ["Victoria, Canada", "Brooklyn, New York"]),
                ("adversarial", "Austin Rivers told Rivers United in Paris, Texas that Paris Long would lead the audit.", ["Austin Rivers", "Paris Long"], ["Rivers United"], ["Paris, Texas"]),
            ],
        ),
        (
            "ner_nested_location_names",
            [
                ("hard", "Mina Patel left the University of Washington for a meeting in Tacoma.", ["Mina Patel"], ["University of Washington"], ["Tacoma"]),
                ("hard", "Leo Martins represented the Bank of Montreal at a hearing in Ottawa.", ["Leo Martins"], ["Bank of Montreal"], ["Ottawa"]),
                ("hard", "Aisha Bello joined the University of Lagos before relocating to Abuja.", ["Aisha Bello"], ["University of Lagos"], ["Abuja"]),
            ],
        ),
        (
            "ner_alias_and_acronym",
            [
                ("hard", "Elena Rossi said the European Space Agency (ESA) would meet in Darmstadt.", ["Elena Rossi"], ["European Space Agency", "ESA"], ["Darmstadt"]),
                ("hard", "Noah Kim joined the International Renewable Energy Agency (IRENA) in Abu Dhabi.", ["Noah Kim"], ["International Renewable Energy Agency", "IRENA"], ["Abu Dhabi"]),
                ("hard", "Fatima Zahra presented for the African Development Bank (AfDB) in Abidjan.", ["Fatima Zahra"], ["African Development Bank", "AfDB"], ["Abidjan"]),
            ],
        ),
        (
            "ner_unicode_and_punctuation",
            [
                ("hard", "Zoë D'Arcy of L'Atelier Nord arrived in São Luís.", ["Zoë D'Arcy"], ["L'Atelier Nord"], ["São Luís"]),
                ("hard", "Søren O'Neill represented Ægir Labs in Tromsø.", ["Søren O'Neill"], ["Ægir Labs"], ["Tromsø"]),
                ("hard", "Ana-María Núñez met Q&R Partners in A Coruña.", ["Ana-María Núñez"], ["Q&R Partners"], ["A Coruña"]),
            ],
        ),
        (
            "ner_multiple_sentences_coreference",
            [
                ("adversarial", "Orion Systems hired Maya Stone in Denver. Stone later opened Orion's office in Lima.", ["Maya Stone"], ["Orion Systems", "Orion"], ["Denver", "Lima"]),
                ("adversarial", "Nexus Health sent Pavel Morozov to Riga. Morozov then represented Nexus in Tallinn.", ["Pavel Morozov"], ["Nexus Health", "Nexus"], ["Riga", "Tallinn"]),
                ("adversarial", "Cedar Bank appointed Lin Wei in Taipei. Wei later spoke for Cedar in Manila.", ["Lin Wei"], ["Cedar Bank", "Cedar"], ["Taipei", "Manila"]),
            ],
        ),
        (
            "ner_distractor_titles_and_products",
            [
                ("adversarial", "Professor Amara Okeke demonstrated the Atlas device for Meridian Robotics in Nairobi.", ["Amara Okeke"], ["Meridian Robotics"], ["Nairobi"]),
                ("adversarial", "Captain Emil Novak tested the Horizon prototype at Solstice Aerospace in Prague.", ["Emil Novak"], ["Solstice Aerospace"], ["Prague"]),
                ("adversarial", "Doctor Hana Suzuki reviewed the Beacon platform for Northstar Medical in Osaka.", ["Hana Suzuki"], ["Northstar Medical"], ["Osaka"]),
            ],
        ),
    ]
    families: list[CandidateFamily] = []
    for family_id, examples in specifications:
        rendered = tuple(
            (
                difficulty,
                _ner_prompt(sentence),
                _ner_truth(persons, organizations, locations),
            )
            for difficulty, sentence, persons, organizations, locations in examples
        )
        families.append(CandidateFamily("ner", family_id, rendered))
    return families


def _summary_prompt(passage: str) -> str:
    return f"Summarize the following passage in 1-2 sentences:\n\n{passage}"


def _summary_family(
    family_id: str,
    rows: list[tuple[str, str, str]],
) -> CandidateFamily:
    return CandidateFamily(
        "summarization",
        family_id,
        tuple((difficulty, _summary_prompt(passage), truth) for difficulty, passage, truth in rows),
    )


def _summarization_families() -> list[CandidateFamily]:
    return [
        _summary_family(
            "summary_reversal_after_review",
            [
                ("hard", "The council first selected the river site for the clinic. A soil review found contamination, so it chose the hill site instead, adding $1.4 million and delaying opening by six months. The vote was 7-2.", "Key facts: council reversed from river site to hill site because of soil contamination; hill site adds $1.4 million and six months; vote was 7-2."),
                ("hard", "The board initially approved Vendor A. After a security audit exposed unpatched systems, it awarded the contract to Vendor C, despite a 12% higher bid and a four-month migration. The decision was 5-1.", "Key facts: board reversed from Vendor A to Vendor C because of security findings; Vendor C costs 12% more and needs four months; vote was 5-1."),
                ("hard", "Planners originally routed the rail line west. New habitat data showed nesting grounds there, so the final route went east, costing $8 million more and opening one year later. The motion passed 6-3.", "Key facts: rail route changed from west to east because of nesting-ground data; east costs $8 million more and opens one year later; vote was 6-3."),
            ],
        ),
        _summary_family(
            "summary_preliminary_vs_final",
            [
                ("hard", "A pilot of 80 patients suggested a 24% reduction in symptoms. The 1,200-patient trial found only 6%, with no statistical significance. The sponsor ended development but will publish the data.", "Key facts: pilot showed 24% reduction; large trial showed 6% and was not significant; development ended; data will be published."),
                ("hard", "Early testing estimated the filter removed 93% of pollutants. A year-long field study measured 61%, below the 75% requirement. The agency rejected certification and requested a redesign.", "Key facts: early estimate was 93%; field result was 61%, below 75%; certification was rejected; redesign requested."),
                ("hard", "The prototype appeared to cut energy use by 18%. Independent replication found a 3% reduction within the margin of error. The company withdrew its savings claim but kept the product on sale.", "Key facts: prototype suggested 18% savings; replication found 3% within error; company withdrew the savings claim but continues sales."),
            ],
        ),
        _summary_family(
            "summary_conditional_exception",
            [
                ("adversarial", "Employees must return to the office three days weekly from May 1. Staff hired as permanently remote before January remain exempt, but new remote hires need quarterly approval. The policy will be reviewed in October.", "Key facts: three office days start May 1; pre-January permanently remote staff are exempt; new remote hires need quarterly approval; review is in October."),
                ("adversarial", "The city bans watering lawns after June 15, except gardens using captured rainwater. Commercial nurseries may request 30-day waivers, while washing vehicles remains prohibited for everyone.", "Key facts: lawn watering is banned after June 15; rainwater gardens are exempt; nurseries can request 30-day waivers; vehicle washing remains prohibited."),
                ("adversarial", "Applications close Friday at noon. Candidates already awaiting accessibility documents get until Tuesday, but only if they notified the office by Wednesday. No extension applies to recommendation letters.", "Key facts: normal deadline is Friday noon; qualifying accessibility-document cases get until Tuesday if notice was given by Wednesday; recommendation letters receive no extension."),
            ],
        ),
        _summary_family(
            "summary_mixed_metrics",
            [
                ("hard", "Revenue rose 14% to $6.2 million and subscriptions grew 21%, but renewals fell from 88% to 79%. Costs increased 19%, reducing operating margin from 16% to 11%.", "Key facts: revenue rose 14% to $6.2M; subscriptions grew 21%; renewals fell from 88% to 79%; costs rose 19%; margin fell from 16% to 11%."),
                ("hard", "Ridership increased 9% and on-time arrivals improved to 92%. Complaints doubled to 1,400, however, and maintenance spending rose 17%, leaving the operating deficit unchanged.", "Key facts: ridership rose 9%; on-time arrivals reached 92%; complaints doubled to 1,400; maintenance spending rose 17%; deficit was unchanged."),
                ("hard", "The school raised graduation to 86% and absenteeism dropped 4 points. Math scores fell 7 points, teacher vacancies reached 12%, and the budget surplus narrowed from $3 million to $800,000.", "Key facts: graduation reached 86%; absenteeism fell 4 points; math scores fell 7 points; vacancies reached 12%; surplus shrank from $3M to $800,000."),
            ],
        ),
        _summary_family(
            "summary_correction_and_retraction",
            [
                ("adversarial", "The report initially said 4,800 homes lost power. The utility corrected that figure to 1,800, explaining that duplicate meter records caused the error. Restoration still finished at 9 PM Monday.", "Key facts: outage count was corrected from 4,800 to 1,800 because of duplicate meter records; restoration finished at 9 PM Monday."),
                ("adversarial", "Officials first announced the bridge would reopen Thursday. They later corrected the date to Tuesday after inspections finished early; the weight restriction remains in force through August.", "Key facts: reopening was corrected from Thursday to Tuesday because inspections finished early; weight restriction lasts through August."),
                ("adversarial", "A journal notice originally identified contamination in 11 samples. A recount found 7 contaminated samples and 4 labeling errors. The paper's main conclusion was unchanged.", "Key facts: contamination count was corrected from 11 to 7; four cases were labeling errors; the paper's main conclusion stayed unchanged."),
            ],
        ),
        _summary_family(
            "summary_dependency_sequence",
            [
                ("hard", "The launch can occur only after the regulator approves the license. Approval depends on a successful evacuation drill scheduled for April 8; if it fails, the drill moves to May and launch slips from June to August.", "Key facts: launch requires license approval; approval requires the April 8 evacuation drill; failure moves the drill to May and launch from June to August."),
                ("hard", "Construction starts once financing closes. Closing requires the environmental permit expected July 3; an appeal would delay the permit 60 days and move completion from March to June.", "Key facts: construction requires financing close; closing requires the July 3 permit; an appeal adds 60 days and shifts completion from March to June."),
                ("hard", "Data migration follows completion of the backup audit. The audit depends on vendor logs due September 10; missing logs trigger a six-week review and postpone the December shutdown until February.", "Key facts: migration requires the backup audit; audit needs vendor logs by September 10; missing logs cause a six-week review and move shutdown from December to February."),
            ],
        ),
    ]


def candidate_families() -> list[CandidateFamily]:
    """Return the versioned, auditable family bank."""
    families = _ner_families() + _summarization_families()
    identifiers = [family.family_id for family in families]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("candidate family IDs must be globally unique")
    return families


def _category_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_family_splits(
    families: list[CandidateFamily],
    *,
    seed: int,
    stress_fraction: float,
) -> dict[str, str]:
    """Assign whole families, stratified by category, before any observations."""
    if not 0.0 < stress_fraction < 1.0:
        raise ValueError("stress_fraction must be strictly between 0 and 1")
    by_category: dict[str, list[str]] = {}
    for family in families:
        by_category.setdefault(family.category, []).append(family.family_id)

    assignments: dict[str, str] = {}
    for category, family_ids in sorted(by_category.items()):
        if len(family_ids) < 2:
            raise ValueError(f"category {category!r} needs at least two families")
        shuffled = sorted(family_ids)
        random.Random(_category_seed(seed, category)).shuffle(shuffled)
        stress_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * stress_fraction)))
        stress_ids = set(shuffled[:stress_count])
        assignments.update(
            {
                family_id: (
                    "stress_holdout" if family_id in stress_ids else "train"
                )
                for family_id in family_ids
            }
        )
    return assignments


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_holdout(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def build_candidate_partitions(
    *,
    seed: int = 31,
    stress_fraction: float = 1 / 3,
    mining_round: int = 1,
    raw_references: list[dict] | None = None,
    holdout_references: list[dict] | None = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Build and validate deterministic train/stress candidate records."""
    if mining_round < 1:
        raise ValueError("mining_round must be at least 1")
    families = candidate_families()
    assignments = assign_family_splits(
        families, seed=seed, stress_fraction=stress_fraction
    )
    partitions: dict[str, list[dict]] = {"train": [], "stress_holdout": []}
    for family in families:
        split = assignments[family.family_id]
        for index, (difficulty, prompt, truth) in enumerate(family.examples):
            task = TaskExample(
                id=(
                    f"mining_r{mining_round:02d}_{family.category}_"
                    f"{family.family_id}_{index:02d}"
                ),
                category=family.category,
                difficulty_pool=difficulty,
                prompt=prompt,
                ground_truth=truth,
                dataset_split=split,
                prompt_family_id=family.family_id,
                generator_version=GENERATOR_VERSION,
                mining_round=mining_round,
            )
            partitions[split].append(task.model_dump())

    train = partitions["train"]
    stress = partitions["stress_holdout"]
    assert_disjoint_mining_partitions(train, stress)

    raw = _read_jsonl(RAW_TASKS_PATH) if raw_references is None else raw_references
    holdout = (
        _read_holdout(HOLDOUT_TASKS_PATH)
        if holdout_references is None
        else holdout_references
    )
    all_candidates = train + stress
    assert_no_prompt_overlap(all_candidates, raw, reference_name="tasks_raw")
    assert_no_prompt_overlap(all_candidates, holdout, reference_name="holdout_tasks")

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "stress_fraction": stress_fraction,
        "mining_round": mining_round,
        "split_before_model_calls": True,
        "family_assignments": dict(sorted(assignments.items())),
        "counts": {
            "train_records": len(train),
            "stress_records": len(stress),
            "train_families": len({prompt_family_key(row) for row in train}),
            "stress_families": len({prompt_family_key(row) for row in stress}),
        },
        "hashes": {
            "train_dataset_sha256": dataset_hash(train),
            "stress_dataset_sha256": dataset_hash(stress),
            "train_prompt_groups_sha256": stable_json_hash(
                sorted(prompt_group_key(row) for row in train)
            ),
            "stress_prompt_groups_sha256": stable_json_hash(
                sorted(prompt_group_key(row) for row in stress)
            ),
            "train_families_sha256": stable_json_hash(
                sorted({prompt_family_key(row) for row in train})
            ),
            "stress_families_sha256": stable_json_hash(
                sorted({prompt_family_key(row) for row in stress})
            ),
            "tasks_raw_sha256": dataset_hash(raw),
            "holdout_tasks_sha256": dataset_hash(holdout),
        },
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    return train, stress, manifest


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_candidate_partitions(
    output_dir: Path,
    train: list[dict],
    stress: list[dict],
    manifest: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    """Atomically write the immutable pre-observation split artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "train_candidates.jsonl",
        output_dir / "stress_candidates.jsonl",
        output_dir / "split_manifest.json",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite mining split artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    _atomic_write_jsonl(paths[0], train)
    _atomic_write_jsonl(paths[1], stress)
    temporary = paths[2].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(paths[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-split candidates for active hard-case mining."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--stress-fraction", type=float, default=1 / 3)
    parser.add_argument("--mining-round", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing deterministic artifacts after validation",
    )
    args = parser.parse_args()
    train, stress, manifest = build_candidate_partitions(
        seed=args.seed,
        stress_fraction=args.stress_fraction,
        mining_round=args.mining_round,
    )
    write_candidate_partitions(
        args.output_dir, train, stress, manifest, force=args.force
    )
    print(
        f"Wrote {len(train)} train and {len(stress)} stress candidates to "
        f"{args.output_dir} (no model calls)."
    )


if __name__ == "__main__":
    main()

