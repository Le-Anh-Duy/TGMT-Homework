import json
from pathlib import Path

root = Path(__file__).resolve().parent
nb_dir = root / "Notebooks"

for nb_path in sorted(nb_dir.glob("*.ipynb")):
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n=== {nb_path.name} (skip: {exc}) ===")
        continue

    print(f"\n=== {nb_path.name} ===")
    cells = nb.get("cells", [])

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        has_image = False
        for output in outputs:
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if isinstance(data, dict) and any(
                key in data for key in ("image/png", "image/jpeg", "image/svg+xml")
            ):
                has_image = True
                break

        if not has_image:
            continue

        source_lines = "".join(cell.get("source", [])).strip().splitlines()
        source_preview = " | ".join(source_lines[:3]) if source_lines else "<empty>"

        previous_markdown = ""
        for j in range(i - 1, -1, -1):
            prev = cells[j]
            if prev.get("cell_type") == "markdown":
                previous_markdown = "".join(prev.get("source", [])).strip().replace("\n", " | ")
                break

        print(f"cell {i + 1}: {source_preview[:200]}")
        if previous_markdown:
            print(f"  prev_md: {previous_markdown[:200]}")
        if source_lines:
            print("  source_lines:")
            for line in source_lines[:12]:
                print(f"    {line}")
