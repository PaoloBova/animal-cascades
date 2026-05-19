import json
from pathlib import Path

JSONL_PATH = Path("datasets/animal_cascade.jsonl")
RFP_DIR = Path("scenarios/rfps")


def _load_samples():
    return [json.loads(line) for line in JSONL_PATH.read_text().splitlines() if line.strip()]


def test_dataset_exists():
    assert JSONL_PATH.exists(), f"Run: uv run python scripts/build_dataset.py"


def test_dataset_has_13_samples():
    assert len(_load_samples()) == 13


def test_all_samples_have_id_and_input():
    for sample in _load_samples():
        assert "id" in sample, f"missing 'id': {sample}"
        assert "input" in sample, f"missing 'input': {sample}"
        assert isinstance(sample["input"], str)
        assert len(sample["input"]) > 100, f"input suspiciously short for {sample['id']}"


def test_no_extra_fields():
    for sample in _load_samples():
        assert set(sample.keys()) == {"id", "input"}, f"unexpected fields in {sample['id']}: {sample.keys()}"


def test_ids_match_rfp_filenames():
    expected = {p.stem for p in RFP_DIR.glob("*.txt")}
    actual = {s["id"] for s in _load_samples()}
    assert actual == expected, f"id mismatch — extra: {actual - expected}, missing: {expected - actual}"


def test_input_matches_rfp_file_content():
    for sample in _load_samples():
        rfp_path = RFP_DIR / f"{sample['id']}.txt"
        assert rfp_path.exists(), f"no matching .txt for {sample['id']}"
        assert sample["input"] == rfp_path.read_text().strip()
