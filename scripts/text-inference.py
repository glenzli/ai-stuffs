"""Load this project's saved small text model and tokenizer; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import learning_core as runtime
import text_training_core as core

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()
# Resolve identities against the bundle, so a same-size foreign tokenizer is rejected.
bundle = args.checkpoint.resolve().parent.parent
checksums = dict((name, sha) for sha, name in (line.split("  ", 1) for line in (bundle / "SHA256SUMS").read_text().splitlines()))
for file in (args.checkpoint, args.tokenizer):
    relative = str(file.resolve().relative_to(bundle))
    if hashlib.sha256(file.read_bytes()).hexdigest() != checksums.get(relative):
        raise ValueError(f"File does not match the saved bundle identity: {relative}")
runtime.configure_cpu(1)
saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
tokenizer = json.loads(args.tokenizer.read_text())
if saved["config"]["vocab_size"] != tokenizer["vocab_size"]:
    raise ValueError("checkpoint and tokenizer vocabulary differ")
model = core.make_model(saved["seed"], saved["config"])
model.load_state_dict(saved["model"]); model.eval()
print(json.dumps(core.generate(model, args.prompt, tokenizer), ensure_ascii=False, indent=2))
