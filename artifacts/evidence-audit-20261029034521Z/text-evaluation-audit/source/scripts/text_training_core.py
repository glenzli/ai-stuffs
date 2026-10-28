"""Small natural-text owner: train-only byte BPE, document windows and decoder.

No experiments, downloads or process-wide RNG/thread changes occur on import.
The symbolic learning-core-v1 remains unchanged.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
import learning_core as symbolic

PAD, BOS, EOS = 0, 1, 2


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def merge_pair(tokens, pair, replacement):
    result, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == tuple(pair):
            result.append(replacement); i += 2
        else:
            result.append(tokens[i]); i += 1
    return result


def train_tokenizer(documents, merges=128):
    if not documents or any(doc["split"] != "train" for doc in documents):
        raise ValueError("Tokenizer training accepts train documents only")
    sequences = [[byte + 3 for byte in p["text"].encode("utf8")] for doc in documents for p in doc["paragraphs"]]
    vocabulary = {str(i + 3): bytes([i]).hex() for i in range(256)}
    rules = []
    for _ in range(merges):
        counts = Counter(pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
        if not counts: break
        pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < 2: break
        new_id = 259 + len(rules)
        vocabulary[str(new_id)] = (bytes.fromhex(vocabulary[str(pair[0])]) + bytes.fromhex(vocabulary[str(pair[1])])).hex()
        rules.append({"pair": list(pair), "id": new_id, "train_frequency": frequency})
        sequences = [merge_pair(sequence, pair, new_id) for sequence in sequences]
    return {"schema": "train-only-byte-bpe-v1", "vocab_size": 259 + len(rules), "special": {"pad": PAD, "bos": BOS, "eos": EOS},
            "vocabulary_hex": vocabulary, "merges": rules, "training_documents": [doc["id"] for doc in documents],
            "training_paragraph_digest": digest([p["id"] for doc in documents for p in doc["paragraphs"]]),
            "scope": "UTF-8 bytes + deterministic most-frequent adjacent-pair merges inside training paragraphs; not the GPT-2 regex/pretokenizer or a downloaded tokenizer"}


def encode(text, tokenizer):
    ids = [byte + 3 for byte in text.encode("utf8")]
    for rule in tokenizer["merges"]: ids = merge_pair(ids, rule["pair"], rule["id"])
    return ids


def decode(ids, tokenizer):
    raw = b"".join(bytes.fromhex(tokenizer["vocabulary_hex"][str(i)]) for i in ids if i >= 3)
    return raw.decode("utf8", errors="replace")


def make_windows(corpus, tokenizer, context=96):
    result = {name: [] for name in ("train", "validation", "test")}
    for doc in corpus["documents"]:
        text = "\n".join(p["text"] for p in doc["paragraphs"])
        ids = [BOS] + encode(text, tokenizer) + [EOS]
        for start in range(0, len(ids) - 1, context):
            end = min(start + context + 1, len(ids))
            result[doc["split"]].append({"id": f"{doc['id']}:{start}", "document": doc["id"], "start": start,
                                          "input": ids[start:end - 1], "target": ids[start + 1:end]})
    return result


def collate(records):
    size = max(len(row["input"]) for row in records)
    ids = torch.full((len(records), size), PAD, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(records):
        ids[i, :len(row["input"])] = torch.tensor(row["input"])
        labels[i, :len(row["target"])] = torch.tensor(row["target"])
    return {"input_ids": ids, "labels": labels, "record_ids": [row["id"] for row in records], "tokens": int((labels != -100).sum())}


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, width=96, depth=2, heads=4, context=96):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width, depth=depth, heads=heads, context=context)
        config = symbolic.ModelConfig(width=width, depth=depth, heads=heads, dropout=0, max_seq_len=context, ff_multiplier=4)
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.blocks = nn.ModuleList([symbolic.DecoderBlock(config) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        positions = torch.arange(context).float()[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        pos = torch.zeros(context, width); pos[:, 0::2] = (positions * frequencies).sin(); pos[:, 1::2] = (positions * frequencies).cos()
        self.register_buffer("positions", pos, persistent=True)
        self.apply(symbolic.SmallDecoder._initialize)
        with torch.no_grad(): self.embedding.weight[PAD].zero_()

    def hidden(self, ids):
        if ids.shape[1] > self.config["context"]: raise ValueError("context exceeded")
        mask = ids != PAD
        allowed = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).tril()[None, None] & mask[:, None, None, :]
        value = self.embedding(ids) + self.positions[:ids.shape[1]]
        for block in self.blocks: value, _ = block(value, allowed)
        return self.norm(value)

    def forward(self, ids):
        return self.head(self.hidden(ids))


def make_model(seed, config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TextDecoder(**config)


def loss(model, batch):
    logits = model(batch["input_ids"])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model, records, batch_size=16):
    total, correct, count = 0., 0, 0
    for start in range(0, len(records), batch_size):
        batch = collate(records[start:start + batch_size]); logits = model(batch["input_ids"])
        labels = batch["labels"]; valid = labels != -100
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum").item()
        correct += int(((logits.argmax(-1) == labels) & valid).sum()); count += batch["tokens"]
    return {"ce_nats_per_token": total / count, "teacher_forced_accuracy": correct / count, "supervised_tokens": count,
            "windows": len(records), "scope": "next-token teacher forcing; does not score instruction following or factual correctness"}


@torch.no_grad()
def generate(model, prompt, tokenizer, new_tokens=48):
    tokens = [BOS] + encode(prompt, tokenizer); emitted = []
    for _ in range(new_tokens):
        next_id = int(model(torch.tensor([tokens[-model.config["context"]:]]))[0, -1].argmax())
        emitted.append(next_id); tokens.append(next_id)
        if next_id == EOS: break
    text = decode(emitted, tokenizer)
    return {"prompt": prompt, "tokens": emitted, "text": text, "replacement_characters": text.count("\ufffd"),
            "special_non_eos": sum(i in (PAD, BOS) for i in emitted), "terminated": bool(emitted and emitted[-1] == EOS),
            "policy": "unmasked greedy; 48-token cap; sliding 96-token context resets local positions when cropped; special tokens and invalid UTF-8 replacements are reported"}


class WindowStream:
    def __init__(self, count, seed):
        self.count, self.rng, self.order, self.cursor = count, random.Random(seed), [], 0

    def next_indices(self, batch_size):
        result = []
        while len(result) < batch_size:
            if self.cursor == len(self.order):
                self.order = list(range(self.count)); self.rng.shuffle(self.order); self.cursor = 0
            take = min(batch_size - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + take]); self.cursor += take
        return result

    def state_dict(self):
        return {"count": self.count, "rng": self.rng.getstate(), "order": self.order.copy(), "cursor": self.cursor}

    def load_state_dict(self, value):
        if value["count"] != self.count: raise ValueError("different window population")
        self.rng.setstate(value["rng"]); self.order = value["order"].copy(); self.cursor = value["cursor"]


def state_error(actual, expected):
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or actual.shape != expected.shape: return 1e30
        return float((actual.detach().double() - expected.detach().double()).abs().max()) if actual.numel() else 0.
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys(): return 1e30
        return max((state_error(actual[key], expected[key]) for key in actual), default=0.)
    if isinstance(actual, (tuple, list)):
        if not isinstance(expected, (tuple, list)) or len(actual) != len(expected): return 1e30
        return max((state_error(a, b) for a, b in zip(actual, expected)), default=0.)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)): return abs(actual - expected)
    return 0. if actual == expected else 1e30
