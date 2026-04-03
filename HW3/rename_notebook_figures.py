import csv
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
nb_dir = root / "Notebooks"
fig_root = nb_dir / "figure"
manifest_path = fig_root / "figure_manifest.csv"


def slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def detect_dataset(source_text: str, markdown_text: str) -> str:
    combined = f"{source_text}\n{markdown_text}".lower()
    if "pneumonia" in combined:
        return "pneumoniamnist"
    if "fashion" in combined:
        return "fashion_mnist"
    if "handwritten" in combined or "mnist" in combined:
        return "mnist_handwritten"
    return "unknown_dataset"


def detect_model(source_text: str, notebook_name: str) -> str:
    s = source_text.lower()
    n = notebook_name.lower()

    if "lenetwide_3x3" in s:
        return "lenet_wide_3x3"
    if "lenetwide(" in s:
        return "lenet_wide_5x5"

    if "newer_baseline" in n:
        return "lenet_modernized"
    if "baseline" in n:
        return "lenet_classic"
    return "lenet"


def detect_stage(model_name: str) -> str:
    if model_name == "lenet_classic":
        return "stage1_classic"
    if model_name == "lenet_modernized":
        return "stage1_modernized"
    if model_name in {"lenet_wide_5x5", "lenet_wide_3x3"}:
        return "stage2_advanced"
    return "stage_unknown"


rows = []
renamed = 0
skipped_notebooks = []

for nb_path in sorted(nb_dir.glob("*.ipynb")):
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        skipped_notebooks.append((nb_path.name, str(exc)))
        continue

    nb_fig_dir = fig_root / nb_path.stem
    nb_fig_dir.mkdir(parents=True, exist_ok=True)

    cells = nb.get("cells", [])

    for ci, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        source_text = "".join(cell.get("source", []))

        prev_md = ""
        for j in range(ci - 2, -1, -1):
            c = cells[j]
            if c.get("cell_type") == "markdown":
                prev_md = "".join(c.get("source", []))
                break

        dataset = detect_dataset(source_text, prev_md)
        model = detect_model(source_text, nb_path.name)
        stage = detect_stage(model)

        image_seq = 0
        for oi, output in enumerate(outputs, start=1):
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if not isinstance(data, dict):
                continue

            ext = None
            if "image/png" in data:
                ext = "png"
            elif "image/jpeg" in data:
                ext = "jpg"
            elif "image/svg+xml" in data:
                ext = "svg"

            if not ext:
                continue

            image_seq += 1
            if image_seq == 1:
                fig_type = "training_history"
            elif image_seq == 2:
                fig_type = "confusion_matrix"
            else:
                fig_type = f"figure_{image_seq}"

            old_name = f"cell_{ci:03d}_output_{oi:02d}.{ext}"
            old_path = nb_fig_dir / old_name
            if not old_path.exists():
                continue

            new_name = f"{stage}_{model}_{dataset}_{fig_type}.{ext}"
            new_path = nb_fig_dir / new_name

            if new_path.exists() and new_path != old_path:
                base = new_name.rsplit(".", 1)[0]
                suffix = 2
                while True:
                    candidate = nb_fig_dir / f"{base}_v{suffix}.{ext}"
                    if not candidate.exists():
                        new_path = candidate
                        break
                    suffix += 1

            old_path.rename(new_path)
            renamed += 1
            rows.append(
                {
                    "notebook": nb_path.name,
                    "cell": ci,
                    "output": oi,
                    "old_file": old_name,
                    "new_file": new_path.name,
                    "stage": stage,
                    "model": model,
                    "dataset": dataset,
                    "figure_type": fig_type,
                }
            )

with manifest_path.open("w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "notebook",
        "cell",
        "output",
        "old_file",
        "new_file",
        "stage",
        "model",
        "dataset",
        "figure_type",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Renamed files: {renamed}")
print(f"Manifest: {manifest_path.as_posix()}")
print(f"Skipped notebooks: {len(skipped_notebooks)}")
for name, err in skipped_notebooks:
    print(f"SKIPPED {name} :: {err}")
