"""Build attributed educational examples from public benchmark samples.

These examples are selected for demonstration and must not be used as an accuracy test.
"""

import io
import json
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    samples = ROOT / "samples"
    samples.mkdir(exist_ok=True)
    manifest = []
    with zipfile.ZipFile(ROOT / "data" / "GTSRB_Final_Test_Images.zip") as archive:
        index = {Path(name).name: name for name in archive.namelist() if name.endswith(".ppm")}
        selections = [
            ("directions.jpg", "Left / straight / right", ["02794.ppm", "12231.ppm", "10131.ppm"]),
            ("combined-directions.jpg", "Combined directions", ["01184.ppm", "08045.ppm"]),
            ("regulatory.jpg", "Stop / no entry", ["09969.ppm", "00144.ppm"]),
        ]
        for filename, label, files in selections:
            canvas = Image.new("RGB", (len(files) * 260, 300), (242, 242, 234))
            for number, file in enumerate(files):
                source = Image.open(io.BytesIO(archive.read(index[file]))).convert("RGB")
                source.thumbnail((220, 220))
                # The source benchmark crops are small; upscale for a legible teaching board.
                scale = min(220 / source.width, 220 / source.height)
                source = source.resize((round(source.width * scale), round(source.height * scale)))
                canvas.paste(source, (number * 260 + (260 - source.width) // 2,
                                      (300 - source.height) // 2))
            canvas.save(samples / filename, quality=94)
            manifest.append({"file": filename, "label": label, "mode": "scene",
                             "description": "Teaching board made from GTSRB photographs; not a road-scene benchmark.",
                             "source": "https://benchmark.ini.rub.de/gtsrb_dataset.html",
                             "source_files": files})
    with tarfile.open(ROOT / "data" / "traffic_light_sample_dataset.tar.gz") as archive:
        for color, number in (("red", 13), ("yellow", 69), ("green", 75)):
            name = f"traffic_light_sample_dataset/val/{color}/{number}.jpg"
            image = Image.open(archive.extractfile(name)).convert("RGB")
            scale = min(500 / image.width, 300 / image.height)
            image = image.resize((round(image.width * scale), round(image.height * scale)))
            filename = f"signal-{color}.jpg"
            image.save(samples / filename, quality=95)
            manifest.append({"file": filename, "label": f"{color.capitalize()} signal", "mode": "signal",
                             "description": "TIER IV public tutorial signal crop. Use close-up signal mode.",
                             "source": "https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz",
                             "source_file": name})
    (samples / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
