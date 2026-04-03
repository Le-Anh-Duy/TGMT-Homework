import base64
import json
from pathlib import Path


root = Path(__file__).resolve().parent
nb_dir = root / "Notebooks"
out_root = nb_dir / "figure"
out_root.mkdir(parents=True, exist_ok=True)

nb_files = sorted(p for p in nb_dir.glob("*.ipynb") if p.is_file())

saved = []
skipped = []
for nb_path in nb_files:
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as exc:
        skipped.append((nb_path, str(exc)))
        continue

    nb_out_dir = out_root / nb_path.stem
    nb_out_dir.mkdir(parents=True, exist_ok=True)

    cells = nb.get("cells", [])
    for ci, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        for oi, out in enumerate(outputs, start=1):
            data = out.get("data", {}) if isinstance(out, dict) else {}
            if not isinstance(data, dict):
                continue

            def norm(value):
                if isinstance(value, list):
                    return "".join(value)
                return value

            png = norm(data.get("image/png"))
            jpg = norm(data.get("image/jpeg"))
            svg = norm(data.get("image/svg+xml"))

            if png:
                fp = nb_out_dir / f"cell_{ci:03d}_output_{oi:02d}.png"
                fp.write_bytes(base64.b64decode(png))
                saved.append(fp)
            elif jpg:
                fp = nb_out_dir / f"cell_{ci:03d}_output_{oi:02d}.jpg"
                fp.write_bytes(base64.b64decode(jpg))
                saved.append(fp)
            elif svg:
                fp = nb_out_dir / f"cell_{ci:03d}_output_{oi:02d}.svg"
                fp.write_text(svg, encoding="utf-8")
                saved.append(fp)

print(f"Notebook files scanned: {len(nb_files)}")
print(f"Figures saved: {len(saved)}")
print(f"Notebook files skipped: {len(skipped)}")
for p in saved:
    print(p.as_posix())
for p, err in skipped:
    print(f"SKIPPED {p.as_posix()} :: {err}")
