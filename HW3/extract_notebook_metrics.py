import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
nb_dir = root / "Notebooks"

metric_pattern = re.compile(r"(accuracy|f1[- ]?score|recall|precision)\s*[:=]\s*([0-9]*\.?[0-9]+%?)", re.IGNORECASE)

for nb_path in sorted(nb_dir.glob("*.ipynb")):
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n=== {nb_path.name} (skip: {exc}) ===")
        continue

    print(f"\n=== {nb_path.name} ===")
    cells = nb.get("cells", [])

    for idx, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        source_l = source.lower()
        if not any(k in source_l for k in ["evaluate_model", "training", "pneumonia", "fashion", "mnist"]):
            continue

        output_texts = []
        for out in cell.get("outputs", []):
            if not isinstance(out, dict):
                continue

            if out.get("output_type") == "stream":
                txt = out.get("text", "")
                if isinstance(txt, list):
                    txt = "".join(txt)
                output_texts.append(txt)

            data = out.get("data", {})
            if isinstance(data, dict):
                for key in ["text/plain", "text/markdown"]:
                    if key in data:
                        txt = data[key]
                        if isinstance(txt, list):
                            txt = "".join(txt)
                        output_texts.append(str(txt))

        if not output_texts:
            continue

        merged = "\n".join(output_texts)
        matches = metric_pattern.findall(merged)

        if matches or any(k in merged.lower() for k in ["accuracy", "f1", "recall", "precision"]):
            src_preview = " | ".join(source.strip().splitlines()[:2])
            print(f"\ncell {idx}: {src_preview[:160]}")
            print("-- raw output snippet --")
            lines = [ln for ln in merged.splitlines() if re.search(r"accuracy|f1|recall|precision|mnist|pneumonia|fashion", ln, re.IGNORECASE)]
            for ln in lines[:20]:
                print(ln)
            if matches:
                print("-- parsed metrics --")
                for name, value in matches:
                    print(f"{name}: {value}")
