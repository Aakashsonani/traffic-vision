"""Extract labelled public test images without selecting for model performance.

Requires the three original archives listed in docs/VALIDATION.md under data/.
The pack is for manual checks, not an independent real-road accuracy benchmark.
"""

import csv
import html
import io
import json
import random
import sys
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_vision.catalogue import GTSRB_LABELS, sign_info  # noqa: E402


def main():
    destination = ROOT / "samples" / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261019)
    excluded_signs, excluded_signals = set(), set()
    demos = json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))
    for item in demos:
        excluded_signs.update(item.get("source_files", []))
        if item.get("source_file"):
            excluded_signals.add(item["source_file"])
    prior_path = ROOT / "outputs" / "pretrained-check.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        excluded_signs.update(row["file"] for row in prior["sign_predictions"])
        excluded_signals.update(row["file"] for row in prior["lamp_predictions"])

    records, evaluation = [], []
    with zipfile.ZipFile(ROOT / "data" / "GTSRB_Final_Test_GT.zip") as archive:
        name = next(name for name in archive.namelist() if name.endswith(".csv"))
        rows = list(csv.DictReader(io.StringIO(archive.read(name).decode()), delimiter=";"))
    with zipfile.ZipFile(ROOT / "data" / "GTSRB_Final_Test_Images.zip") as archive:
        index = {Path(name).name: name for name in archive.namelist() if name.endswith(".ppm")}
        for class_id, raw_label in enumerate(GTSRB_LABELS):
            info = sign_info(raw_label)
            candidates = sorted((row for row in rows if int(row["ClassId"]) == class_id
                                 and row["Filename"] not in excluded_signs),
                                key=lambda row: row["Filename"])
            for number, row in enumerate(rng.sample(candidates, 2), 1):
                filename = f"sign-{class_id:02d}-{info['id']}-{number}.png"
                image = Image.open(io.BytesIO(archive.read(index[row["Filename"]]))).convert("RGB")
                image.save(destination / filename)
                box = [int(row[key]) for key in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]
                records.append({"file": filename, "mode": "Full road scene",
                                "expected_label": info["label"], "expected_id": info["id"],
                                "expected_color": "", "bbox": ",".join(map(str, box)),
                                "source": "GTSRB official test", "source_file": row["Filename"]})
                evaluation.append({"file": filename, "group": "gtsrb-official-test",
                                   "split": "test", "objects": [{"label": info["id"], "bbox": box}]})

    with tarfile.open(ROOT / "data" / "traffic_light_sample_dataset.tar.gz") as archive:
        for color in ("red", "yellow", "green"):
            candidates = sorted(name for name in archive.getnames()
                                if f"/val/{color}/" in name and name.endswith(".jpg")
                                and name not in excluded_signals)
            for number, name in enumerate(rng.sample(candidates, 6), 1):
                filename = f"signal-{color}-{number}.jpg"
                # Preserve the original JPEG bytes; no resizing, enhancement or re-encoding.
                (destination / filename).write_bytes(archive.extractfile(name).read())
                records.append({"file": filename, "mode": "Close-up traffic signal",
                                "expected_label": f"{color.capitalize()} signal (colour check only)",
                                "expected_id": "", "expected_color": color, "bbox": "",
                                "source": "TIER IV tutorial validation", "source_file": name})

    with (destination / "expected-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (destination / "sign-evaluation.json").write_text(json.dumps({
        "name": "Public GTSRB manual-check pack",
        "scope": "86 public sign crops, 43 classes; upstream training overlap unknown; not road-scene accuracy",
        "images": evaluation,
    }, indent=2) + "\n", encoding="utf-8")
    (destination / "README.txt").write_text(
        "TRAFFIC VISION - SAMPLE IMAGE PACK\n\n"
        "104 real image crops: 86 signs (two per GTSRB class) and 18 signal crops.\n"
        "Open index.html for a labelled gallery; expected-results.csv is the answer key.\n\n"
        "HOW TO TEST\n"
        "1. Open http://127.0.0.1:8765 and upload an image.\n"
        "2. Use Full road scene for sign-*.png.\n"
        "3. Use Close-up traffic signal for signal-*.jpg.\n"
        "4. Start at the default 55% minimum score. Compare with the answer key.\n"
        "5. Record incorrect, uncertain and missed observations, including high-score errors.\n\n"
        "These images were sampled without running the app on them or selecting for success.\n"
        "Previously used demo/check images were excluded. Upstream model training overlap\n"
        "is unknown, so this is not an independent real-road accuracy benchmark. The set\n"
        "contains small crops, not complete road scenes. Signal labels establish colour\n"
        "only, not arrow direction. There are no verified arrow-lamp or urban no-turn\n"
        "prohibition examples in this pack; collect separate labelled images for those.\n\n"
        "OPTIONAL SIGN OBJECT EVALUATION (from the project folder):\n"
        "python scripts/evaluate.py --dataset samples/validation/sign-evaluation.json "
        "--output outputs/sample-pack-evaluation.json\n"
        "The bounding boxes in that file are from the original GTSRB annotations.\n"
        "This does not replace the local-road Evaluation page report.\n\n"
        "SOURCES / ATTRIBUTION\n"
        "Signs: German Traffic Sign Recognition Benchmark, J. Stallkamp et al.\n"
        "https://benchmark.ini.rub.de/\n"
        "Signal crops: TIER IV / Autoware public tutorial dataset\n"
        "https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz\n"
        "Original image rights remain with their authors. Educational testing examples.\n",
        encoding="utf-8")
    cards = "\n".join(
        f'<a class="card" href="{html.escape(row["file"])}" download>'
        f'<div class="photo"><img src="{html.escape(row["file"])}" alt="{html.escape(row["expected_label"])}" loading="lazy"></div>'
        f'<strong>{html.escape(row["expected_label"])}</strong>'
        f'<small>{html.escape(row["mode"])}</small>'
        f'<code>{html.escape(row["file"])}</code></a>' for row in records)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic Vision - Sample images and expected labels</title>
<style>body{font:15px/1.7 system-ui,sans-serif;color:#28312b;background:#f6f5f1;margin:0;padding:32px;max-width:1400px;margin:auto}
h1{font:40px Georgia,serif;margin:12px 0}p{max-width:900px;color:#626e5c}a{color:#355b3e}nav{display:flex;gap:24px;margin:20px 0 30px;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}.card{background:white;border:1px solid #eaeaea;border-radius:8px;padding:16px;text-decoration:none}
.photo{height:130px;display:flex;align-items:center;justify-content:center;background:#f4f5ef;margin-bottom:14px}.photo img{max-width:100%;max-height:120px;image-rendering:auto}
strong,small,code{display:block}small{font-size:11px;color:#75816e}code{font-size:10px;overflow-wrap:anywhere;margin-top:8px;color:#8b9683}
.note{padding:16px 20px;background:#ebf0e5;border:1px solid #dce4d3;border-radius:6px;margin:24px 0}a:focus-visible{outline:3px solid #718e62;outline-offset:3px}
@media(max-width:500px){body{padding:20px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.card{padding:10px}h1{font-size:32px}}</style>
<h1>Sample images. Expected meanings.</h1>
<p>104 real image crops: two examples of each of 43 sign classes, plus six examples of each signal colour. Click an image to download it, then upload it to Traffic Vision.</p>
<nav><a href="http://127.0.0.1:8765">Open Traffic Vision</a><a href="expected-results.csv" download>Download answer key (CSV)</a><a href="../traffic-vision-validation-pack.zip" download>Download all images (ZIP)</a><a href="README.txt">Testing instructions</a></nav>
<div class="note"><b>Choose the matching mode.</b> Signs: Full road scene. Lights: Close-up traffic signal. Start at 55% score. These are public crops with unknown upstream training overlap, so this pack cannot establish accuracy on local road scenes. Arrow-lamp directions are not labelled in this pack.</div>
<div class="grid">""" + cards + "</div></html>\n"
    (destination / "index.html").write_text(page, encoding="utf-8")
    archive_path = ROOT / "samples" / "traffic-vision-validation-pack.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.iterdir()):
            archive.write(path, arcname="traffic-vision-samples/" + path.name)
    print(f"Created {len(records)} images and answer key: {archive_path}")


if __name__ == "__main__":
    main()
