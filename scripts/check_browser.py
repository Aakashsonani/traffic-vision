"""Exercise real inference, exports, mobile layout, video and a synthetic webcam.

Starts an isolated temporary local server. Requires the dev extra and installed
Edge/Chrome (or `playwright install chromium`). Exits nonzero on any failed check.
"""

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (ROOT / "outputs" / "browser-server.log").open("w") as logs:
        server = subprocess.Popen([sys.executable, "-m", "traffic_vision", "--port", str(port)],
                                  cwd=ROOT, stdout=logs, stderr=logs,
                                  creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(base + "/api/health", timeout=1) as response:
                        assert json.load(response)["ready"], "Run setup.ps1 to install all models"
                    break
                except OSError:
                    if server.poll() is not None:
                        raise RuntimeError("Test server stopped; see outputs/browser-server.log")
                    time.sleep(.2)
            else:
                raise RuntimeError("Test server did not become ready")
            with sync_playwright() as playwright:
                options = {"headless": True, "args": ["--use-fake-ui-for-media-stream",
                           "--use-fake-device-for-media-stream"]}
                browser = None
                for channel in ("msedge", "chrome", None):
                    try:
                        browser = playwright.chromium.launch(**options, **({"channel": channel} if channel else {}))
                        break
                    except Exception:
                        continue
                if browser is None:
                    raise RuntimeError("Install Edge/Chrome or run: playwright install chromium")
                context = browser.new_context(viewport={"width": 1512, "height": 1050},
                                              permissions=["camera"], accept_downloads=True)
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base)
                expect(page.locator("#model-status")).to_have_text("Models ready · CPU")
                expect(page.locator("#analyze-button")).to_be_disabled()
                page.locator("[data-sample='0']").click()
                page.locator("#analyze-button").click()
                expect(page.locator(".observation")).to_have_count(3, timeout=90000)
                for label in ("Turn left ahead", "Straight ahead", "Turn right ahead"):
                    expect(page.locator(".obs-title").filter(has_text=label)).to_be_visible()
                with page.expect_download() as result:
                    page.locator("#download-json").click()
                payload = json.loads(Path(result.value.path()).read_text(encoding="utf-8"))
                assert {d["id"] for d in payload["detections"]} == {"turn_left", "straight", "turn_right"}
                page.locator("#toggle-original").click()
                assert page.locator("#preview-image").get_attribute("src").startswith("blob:")
                page.locator("#toggle-original").click()
                if args.screenshots:
                    folder = ROOT / "docs" / "screenshots"
                    folder.mkdir(exist_ok=True)
                    page.evaluate("window.scrollTo(0,0); document.getElementById('toast').classList.add('hidden')")
                    page.screenshot(path=str(folder / "studio.png"), full_page=True)
                page.locator("[data-page='library']").click()
                page.locator("#search-signs").fill("no left")
                expect(page.locator(".sign-card")).to_have_count(1)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile overflow"
                page.locator("[data-page='evaluation']").click()
                expect(page.locator("#evaluation-content")).to_contain_text("not been measured")
                page.locator("[data-page='analyzer']").click()
                page.set_viewport_size({"width": 1512, "height": 1050})
                # Build an actual short WebM using the browser's own supported codec.
                encoded = page.evaluate("""async () => {
                    const image = new Image(); image.src = '/samples/directions.jpg'; await image.decode();
                    const canvas = document.createElement('canvas'); canvas.width=780; canvas.height=300;
                    canvas.getContext('2d').drawImage(image,0,0,780,300);
                    const stream = canvas.captureStream(5);
                    const recorder = new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp8'});
                    const chunks=[]; recorder.ondataavailable=(e)=>chunks.push(e.data);
                    const stopped=new Promise(r=>recorder.onstop=r); recorder.start();
                    await new Promise(r=>setTimeout(r,1300)); recorder.stop(); await stopped;
                    stream.getTracks().forEach(t=>t.stop());
                    const blob=new Blob(chunks,{type:'video/webm'});
                    return await new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(blob);});
                }""")
                video = ROOT / "outputs" / "browser-video.webm"
                video.write_bytes(base64.b64decode(encoded))
                page.locator("#file-input").set_input_files(str(video))
                page.locator("#analyze-button").click()
                expect(page.locator("#analyze-button")).to_be_enabled(timeout=90000)
                with page.expect_download() as result:
                    page.locator("#download-json").click()
                video_report = json.loads(Path(result.value.path()).read_text(encoding="utf-8"))
                assert len(video_report["frames"]) >= 1
                page.locator("[data-source='camera']").click()
                page.locator("#enable-camera").click()
                expect(page.locator("#enable-camera")).to_have_text("Camera enabled", timeout=15000)
                page.locator("#analyze-button").click()
                expect(page.locator("#download-json")).to_be_enabled(timeout=90000)
                page.locator("#stop-button").click()
                expect(page.locator("#analyze-button")).to_be_enabled()
                page.locator("[data-source='upload']").click()
                page.locator("#file-input").set_input_files({"name": "broken.jpg", "mimeType": "image/jpeg",
                                                            "buffer": b"invalid"})
                page.locator("#analyze-button").click()
                expect(page.locator("#toast")).to_contain_text("could not be read", timeout=15000)
                assert not errors, errors
                browser.close()
                print("PASS: real sign inference, JSON export, original toggle, catalogue, evaluation, mobile, video, synthetic webcam, invalid upload; no JavaScript errors.")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
