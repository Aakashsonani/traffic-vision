import io

import pytest

from scripts import download_models


def test_checksum_mismatch_does_not_install_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(download_models.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(b"bad"))
    monkeypatch.setattr(download_models.time, "sleep", lambda _: None)
    target = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_models.download("https://example.com/model.onnx", target, "0" * 64)
    assert not target.exists()
    assert not target.with_suffix(".onnx.part").exists()
