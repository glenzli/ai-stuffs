"""Download one pinned public encoder; every actual byte is recorded for replay."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"]


def download(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true&expand=true", timeout=60) as response:
        metadata = {r["path"]: r for r in json.load(response)}
    entries = []
    for name in FILES:
        file = destination / name
        if not file.exists():
            temporary = file.with_suffix(file.suffix + ".partial")
            with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120) as response, temporary.open("wb") as handle:
                while part := response.read(1024 * 1024): handle.write(part)
            temporary.rename(file)
        digest = hashlib.sha256(file.read_bytes()).hexdigest(); expected = metadata[name].get("lfs", {}).get("oid")
        if file.stat().st_size != metadata[name]["size"] or (expected and digest != expected): raise ValueError(f"Model identity mismatch: {name}")
        entries.append({"file": name, "bytes": file.stat().st_size, "sha256": digest, "publisher_lfs_sha256": expected})
        print(f"Verified {name}: {file.stat().st_size} bytes", flush=True)
    manifest = {"model_id": MODEL, "revision": REVISION, "license": "MIT", "source": f"https://huggingface.co/{MODEL}/tree/{REVISION}", "files": entries,
                "policy": "public pinned weights; no trust_remote_code, no service inference, no API key"}
    (destination / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", required=True)
    download(parser.parse_args().out_dir)
