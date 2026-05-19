import json
from pathlib import Path

RFP_DIR = Path("scenarios/rfps")
OUTPUT = Path("datasets/animal_cascade.jsonl")


def build() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    samples = [
        {"id": path.stem, "input": path.read_text().strip()}
        for path in sorted(RFP_DIR.glob("*.txt"))
    ]
    with OUTPUT.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote {len(samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    build()
