"""Shared CPU decoder and reversal task. See docs/learning-core-api.md.

Importing this module does not start experiments or change thread/RNG settings.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import platform
import random
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

CORE_API_VERSION = "learning-core-v1"
PAD, BOS, SEP, EOS = 0, 1, 2, 3
CONTENT_IDS = (4, 5, 6, 7)
VOCAB = ("PAD", "BOS", "SEP", "EOS", "A", "B", "C", "D")
VOCAB_SIZE = len(VOCAB)
IGNORE_INDEX = -100
DEFAULT_MAX_NEW_TOKENS = 20


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    depth: int = 2
    heads: int = 4
    dropout: float = 0.1
    norm: str = "pre"
    residual_scale: float = 1.0
    ff_multiplier: int = 4
    max_seq_len: int = 64
    norm_eps: float = 1e-5

    def __post_init__(self):
        for key in ("width", "depth", "heads", "ff_multiplier", "max_seq_len"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.norm not in {"pre", "post", "none"}:
            raise ValueError("norm must be pre, post or none (LayerNorm)")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def configure_cpu(threads: int = 1, deterministic: bool = True) -> None:
    """Explicit caller opt-in; no process-wide configuration on import."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def environment_metadata() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "device": "cpu", "dtype": "float32",
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_build": torch.__config__.show()}


def _record(content: Sequence[int]) -> dict:
    content = list(content)
    text = "".join(VOCAB[x] for x in content)
    reverse = text[::-1]
    return {"id": f"reverse-{len(content)}-{text}",
            "group_id": f"reversal-{len(content)}-{min(text, reverse)}",
            "content": content, "target": content[::-1], "length": len(content)}


def make_task_splits(data_seed: int = 20261028, *,
                     train_lengths: Sequence[int] = (2, 3, 4, 5),
                     length_lengths: Sequence[int] = (6, 7, 8),
                     train_fraction: float = 0.6,
                     validation_fraction: float = 0.2,
                     max_groups_per_length: int = 180,
                     length_groups_per_length: int = 24) -> dict[str, list[dict]]:
    """Split reversal-equivalence groups *before* training or model selection.

    In-support groups are stratified by length. Entire selected groups are
    retained, so record counts depend on palindromes rather than a row quota.
    No labels or evaluation predictions influence group selection.
    """
    train_lengths, length_lengths = tuple(train_lengths), tuple(length_lengths)
    if (not train_lengths or not length_lengths or
            len(set(train_lengths)) != len(train_lengths) or
            len(set(length_lengths)) != len(length_lengths) or
            set(train_lengths) & set(length_lengths) or
            any(not isinstance(n, int) or n < 1 or n > 10 for n in train_lengths + length_lengths)):
        raise ValueError("lengths must be distinct positive integers <= 10 in disjoint supports")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave positive train, validation and test fractions")
    if max_groups_per_length < 3 or length_groups_per_length < 1:
        raise ValueError("group quotas must be positive (in-support >= 3)")
    rng = random.Random(data_seed)
    splits = {name: [] for name in ("train", "validation", "test", "length")}
    for n in train_lengths + length_lengths:
        quota = max_groups_per_length if n in train_lengths else length_groups_per_length
        # Enumerate short supports; rejection-sample larger supports without
        # materializing 4**n records. Sampling operates only on input groups.
        groups: dict[tuple[int, ...], None] = {}
        if n <= 6:
            keys = [x for x in itertools.product(CONTENT_IDS, repeat=n) if x <= x[::-1]]
            rng.shuffle(keys)
            keys = keys[:quota]
        else:
            available = (4**n + 4**((n + 1)//2)) // 2
            while len(groups) < min(quota, available):
                x = tuple(rng.choice(CONTENT_IDS) for _ in range(n))
                groups[min(x, x[::-1])] = None
            keys = list(groups)
            rng.shuffle(keys)
        if n in train_lengths:
            count = len(keys)
            n_train = max(1, min(count - 2, int(count * train_fraction)))
            n_val = max(1, min(count - n_train - 1, int(count * validation_fraction)))
            allocations = (("train", keys[:n_train]), ("validation", keys[n_train:n_train+n_val]),
                           ("test", keys[n_train+n_val:]))
        else:
            allocations = (("length", keys),)
        for name, selected in allocations:
            for x in selected:
                splits[name].append(_record(x))
                if x != x[::-1]:
                    splits[name].append(_record(x[::-1]))
    assert_split_integrity(splits)
    return splits


def assert_split_integrity(splits: dict[str, list[dict]]) -> dict:
    inputs, groups, ids = set(), set(), set()
    result = {}
    for name, records in splits.items():
        own_inputs, own_groups, own_ids = set(), set(), set()
        for row in records:
            content = tuple(row["content"])
            expected = _record(content)
            if any(row[k] != expected[k] for k in ("id", "group_id", "target", "length")):
                raise ValueError("record violates reversal task contract")
            if content in own_inputs or row["id"] in own_ids:
                raise ValueError("duplicate exact input or ID inside split")
            own_inputs.add(content); own_groups.add(row["group_id"]); own_ids.add(row["id"])
        if inputs & own_inputs or groups & own_groups or ids & own_ids:
            raise ValueError("exact inputs or reversal-equivalence groups leak across splits")
        inputs |= own_inputs; groups |= own_groups; ids |= own_ids
        result[name] = {"records": len(records), "groups": len(own_groups),
                        "lengths": sorted(set(r["length"] for r in records))}
    if "length" in splits:
        supported = {r["length"] for name in ("train", "validation", "test") for r in splits.get(name, [])}
        if supported & {r["length"] for r in splits["length"]}:
            raise ValueError("length extrapolation overlaps training length support")
    return result


def split_digest(splits: dict) -> str:
    return hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collate_records(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("cannot collate an empty batch")
    sequences, labels = [], []
    for row in records:
        content, target = list(row["content"]), list(row["target"])
        if not content or any(x not in CONTENT_IDS for x in content + target):
            raise ValueError("records require nonempty content symbols and content-only target")
        tokens = [BOS] + content + [SEP] + target + [EOS]
        sequences.append(tokens[:-1])
        labels.append([IGNORE_INDEX] * (len(content) + 1) + target + [EOS])
    width = max(map(len, sequences))
    input_ids = torch.full((len(records), width), PAD, dtype=torch.long)
    label_ids = torch.full_like(input_ids, IGNORE_INDEX)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (tokens, targets) in enumerate(zip(sequences, labels)):
        input_ids[i, :len(tokens)] = torch.tensor(tokens)
        label_ids[i, :len(targets)] = torch.tensor(targets)
        attention_mask[i, :len(tokens)] = True
    counts = (label_ids != IGNORE_INDEX).sum(dim=1)
    return {"input_ids": input_ids, "labels": label_ids, "attention_mask": attention_mask,
            "record_ids": [r["id"] for r in records], "ids": [r["id"] for r in records],
            "group_ids": [r["group_id"] for r in records],
            "supervised_tokens": int(counts.sum()), "supervised_per_record": counts,
            "input_tokens": int(attention_mask.sum()), "records": list(records)}


@dataclass
class DecoderCache:
    layers: tuple[tuple[Tensor, Tensor], ...]
    attention_mask: Tensor

    @property
    def length(self) -> int:
        return self.attention_mask.shape[1]


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.head_width = config.heads, config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.dropout = config.dropout

    def forward(self, hidden: Tensor, allowed: Tensor, past=None):
        batch, time, width = hidden.shape
        q, k, v = self.qkv(hidden).view(batch, time, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                               dropout_p=self.dropout if self.training else 0.0)
        return self.output(result.transpose(1, 2).contiguous().view(batch, time, width)), (k, v)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_kind, self.residual_scale = config.norm, config.residual_scale
        self.norm1 = nn.LayerNorm(config.width, eps=config.norm_eps) if config.norm != "none" else nn.Identity()
        self.norm2 = nn.LayerNorm(config.width, eps=config.norm_eps) if config.norm != "none" else nn.Identity()
        self.attention = CausalAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.width, config.ff_multiplier * config.width), nn.GELU(),
                                 nn.Linear(config.ff_multiplier * config.width, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, allowed: Tensor, past=None):
        branch = self.norm1(hidden) if self.norm_kind == "pre" else hidden
        attention, cache = self.attention(branch, allowed, past)
        hidden = hidden + self.residual_scale * self.dropout(attention)
        if self.norm_kind == "post":
            hidden = self.norm1(hidden)
        branch = self.norm2(hidden) if self.norm_kind == "pre" else hidden
        hidden = hidden + self.residual_scale * self.dropout(self.mlp(branch))
        if self.norm_kind == "post":
            hidden = self.norm2(hidden)
        return hidden, cache


class SmallDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.width, padding_idx=PAD)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(config.width, eps=config.norm_eps) if config.norm == "pre" else nn.Identity()
        self.lm_head = nn.Linear(config.width, VOCAB_SIZE, bias=False)
        positions = torch.arange(config.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, config.width, 2, dtype=torch.float32) * (-math.log(10000.0) / config.width))
        encoding = torch.zeros(config.max_seq_len, config.width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:config.width//2])
        self.register_buffer("position_encoding", encoding, persistent=False)
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[PAD].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, input_ids: Tensor, attention_mask: Tensor | None, cache: DecoderCache | None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be nonempty int64 [batch, time]")
        if attention_mask is None:
            attention_mask = input_ids != PAD
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be bool [batch, new_time], True=valid")
        previous = 0 if cache is None else cache.length
        if cache is not None and (len(cache.layers) != len(self.blocks) or cache.attention_mask.shape[0] != input_ids.shape[0]):
            raise ValueError("cache architecture or batch differs")
        complete_mask = attention_mask if cache is None else torch.cat((cache.attention_mask, attention_mask), dim=1)
        if previous + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("physical sequence length exceeds max_seq_len")
        positions = complete_mask.long().cumsum(dim=1)[:, previous:] - 1
        positions = positions.clamp_min(0)
        hidden = self.embedding_dropout(self.token_embedding(input_ids) + self.position_encoding[positions])
        queries = torch.arange(previous, previous + input_ids.shape[1], device=input_ids.device)
        keys = torch.arange(complete_mask.shape[1], device=input_ids.device)
        allowed = (keys[None, :] <= queries[:, None])[None, None, :, :] & complete_mask[:, None, None, :]
        layers = []
        for i, block in enumerate(self.blocks):
            hidden, kv = block(hidden, allowed, None if cache is None else cache.layers[i])
            layers.append(kv)
        return self.lm_head(self.final_norm(hidden)), DecoderCache(tuple(layers), complete_mask)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Ordinary full forward returns logits, never a loss or tuple."""
        return self._forward(input_ids, attention_mask, None)[0]

    def forward_cached(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                       cache: DecoderCache | None = None) -> tuple[Tensor, DecoderCache]:
        """Project only new tokens and append per-layer K/V; eval-only path."""
        if self.training:
            raise ValueError("forward_cached requires eval mode (dropout disabled)")
        return self._forward(input_ids, attention_mask, cache)


def make_model(seed: int = 11, **config) -> SmallDecoder:
    """CPU model, isolated construction RNG; global training RNG is unchanged."""
    config = dict(config)
    norm = config.get("norm")
    if norm in {"pre-ln", "pre_ln", "pre-LN"}:
        config["norm"] = "pre"
    if norm in {"post-ln", "post_ln", "post-LN"}:
        config["norm"] = "post"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SmallDecoder(ModelConfig(**config)).cpu()


def loss_from_logits(logits: Tensor, labels: Tensor, reduction: str = "token") -> Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits/labels shapes differ")
    valid = labels != IGNORE_INDEX
    count = valid.sum(dim=1)
    if not bool(valid.any()):
        raise ValueError("no supervised tokens: caller must skip the update")
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                ignore_index=IGNORE_INDEX, reduction="none").view_as(labels)
    if reduction in {"token", "mean"}:
        return per_token.sum() / count.sum()
    if reduction in {"sequence", "per_sequence"}:
        return (per_token.sum(dim=1)[count > 0] / count[count > 0]).mean()
    if reduction == "sum":
        return per_token.sum()
    raise ValueError("reduction must be token, sequence or sum")


def forward_loss(model: SmallDecoder, batch: dict, reduction: str = "token") -> Tensor:
    if not bool((batch["labels"] != IGNORE_INDEX).any()):
        raise ValueError("no supervised tokens: caller must skip the update")
    return loss_from_logits(model(batch["input_ids"], attention_mask=batch["attention_mask"]), batch["labels"], reduction)


@contextmanager
def evaluation_mode(model: nn.Module):
    modes = {module: module.training for module in model.modules()}
    rng = capture_rng_state()
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        restore_rng_state(rng)


def score_generation(predicted: Sequence[int], target: Sequence[int], *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict:
    """Positionwise score including EOS; extra/missing slots count as errors.

    Target is content-only. The actual emitted sequence includes EOS if emitted.
    Stop at first EOS; tokens supplied after EOS are an invalid trajectory.
    """
    predicted, target = list(predicted), list(target)
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if any(x not in CONTENT_IDS for x in target):
        raise ValueError("target must contain content symbols only")
    if len(predicted) > max_new_tokens or any(x not in range(VOCAB_SIZE) for x in predicted):
        raise ValueError("invalid generated tokens or cap exceeded")
    if EOS in predicted[:-1]:
        raise ValueError("tokens appear after termination")
    gold = target + [EOS]
    slots = max(len(predicted), len(gold))
    correct = sum(int(a == b) for a, b in zip(predicted, gold))
    terminated = bool(predicted and predicted[-1] == EOS)
    content = predicted[:-1] if terminated else predicted
    return {"correct_tokens": correct, "scored_tokens": slots,
            "token_accuracy": correct / slots, "exact_match": bool(terminated and predicted == gold),
            "terminated": terminated, "termination": "eos" if terminated else ("cap" if len(predicted) == max_new_tokens else "incomplete"),
            "early_eos": bool(terminated and len(content) < len(target)),
            "missing_tokens": max(0, len(target) - len(content)),
            "extra_tokens": max(0, len(content) - len(target)),
            "eos_position_correct": bool(terminated and len(content) == len(target))}


def generate_records(model: SmallDecoder, records: Sequence[dict], *,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     use_cache: bool = True) -> list[dict]:
    """Greedy free generation with a single shared cap, never a gold stop."""
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    result = []
    with evaluation_mode(model):
        # Group by *input prompt* length so no PAD is inserted into a cache.
        groups: dict[int, list[tuple[int, dict]]] = {}
        for i, row in enumerate(records):
            groups.setdefault(len(row["content"]), []).append((i, row))
        for n, rows in groups.items():
            if n + 2 + max_new_tokens > model.config.max_seq_len:
                raise ValueError("prompt plus common generation cap exceeds model context")
            tokens = torch.tensor([[BOS] + list(r["content"]) + [SEP] for _, r in rows], dtype=torch.long)
            predictions = [[] for _ in rows]
            finished = torch.zeros(len(rows), dtype=torch.bool)
            valid = torch.ones_like(tokens, dtype=torch.bool)
            logits, cache = model.forward_cached(tokens, attention_mask=valid) if use_cache else (model(tokens, attention_mask=valid), None)
            for step in range(max_new_tokens):
                next_ids = logits[:, -1].argmax(dim=-1)
                for j, next_id in enumerate(next_ids.tolist()):
                    if not finished[j]:
                        predictions[j].append(next_id)
                finished |= next_ids == EOS
                if bool(finished.all()) or step + 1 == max_new_tokens:
                    break
                next_ids = torch.where(finished, torch.full_like(next_ids, EOS), next_ids).unsqueeze(1)
                if use_cache:
                    # Emitted PAD/BOS/SEP are invalid answers but remain actual
                    # positions. Only training/collation PAD is excluded.
                    logits, cache = model.forward_cached(next_ids, attention_mask=torch.ones_like(next_ids, dtype=torch.bool), cache=cache)
                else:
                    tokens = torch.cat((tokens, next_ids), dim=1)
                    logits = model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))
            for (index, row), prediction in zip(rows, predictions):
                result.append({"index": index, "id": row["id"], "group_id": row["group_id"],
                               "content": list(row["content"]), "target": list(row["target"]),
                               "length": row["length"], "predicted": prediction,
                               **score_generation(prediction, row["target"], max_new_tokens=max_new_tokens)})
    return [{k: v for k, v in row.items() if k != "index"} for row in sorted(result, key=lambda x: x["index"])]


def summarize_predictions(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty predictions")
    count = len(rows)
    correct, total = sum(r["correct_tokens"] for r in rows), sum(r["scored_tokens"] for r in rows)
    return {"records": count, "correct_tokens": correct, "scored_tokens": total,
            "free_token_accuracy": correct / total,
            "free_token_accuracy_macro": sum(r["token_accuracy"] for r in rows) / count,
            "exact_match": sum(r["exact_match"] for r in rows) / count,
            "exact_matches": sum(r["exact_match"] for r in rows),
            "eos_rate": sum(r["terminated"] for r in rows) / count,
            "early_eos_count": sum(r["early_eos"] for r in rows),
            "cap_count": sum(r["termination"] == "cap" for r in rows),
            "incomplete_count": sum(r["termination"] == "incomplete" for r in rows),
            "missing_tokens": sum(r["missing_tokens"] for r in rows),
            "extra_tokens": sum(r["extra_tokens"] for r in rows),
            "eos_position_correct": sum(r["eos_position_correct"] for r in rows) / count}


def evaluate_records(model: SmallDecoder, records: Sequence[dict], *, batch_size: int = 64,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     include_predictions: bool = True, use_cache: bool = True) -> dict:
    if not records or batch_size < 1:
        raise ValueError("evaluation requires records and positive batch_size")
    total_loss, teacher_correct, total_tokens = 0.0, 0, 0
    teacher_rows = {}
    with evaluation_mode(model):
        for offset in range(0, len(records), batch_size):
            selected = records[offset:offset+batch_size]
            batch = collate_records(selected)
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            labels = batch["labels"]
            valid = labels != IGNORE_INDEX
            losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                                     ignore_index=IGNORE_INDEX, reduction="none").view_as(labels)
            guesses = logits.argmax(dim=-1)
            total_loss += float(losses.sum())
            teacher_correct += int(((guesses == labels) & valid).sum())
            total_tokens += int(valid.sum())
            for i, row in enumerate(selected):
                teacher_rows[row["id"]] = {"teacher_forced_loss": float(losses[i].sum() / valid[i].sum()),
                                           "teacher_forced_predicted": guesses[i][valid[i]].tolist()}
    predictions = []
    for offset in range(0, len(records), batch_size):
        predictions.extend(generate_records(model, records[offset:offset+batch_size],
                                            max_new_tokens=max_new_tokens, use_cache=use_cache))
    for row in predictions:
        row.update(teacher_rows[row["id"]])
    metrics = {**summarize_predictions(predictions), "teacher_forced_loss": total_loss / total_tokens,
               "teacher_forced_token_accuracy": teacher_correct / total_tokens,
               "supervised_tokens": total_tokens, "max_new_tokens": max_new_tokens,
               "generation": "greedy-kv-cache" if use_cache else "greedy-full-forward"}
    result = {"metrics": metrics, "by_length": {str(n): summarize_predictions([r for r in predictions if r["length"] == n])
                                                for n in sorted({r["length"] for r in predictions})}}
    if include_predictions:
        result["predictions"] = predictions
    return result


def baseline_predictions(records: Sequence[dict], kind: str, *, constant_token: int = 4,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict:
    """Deterministic task-aware baselines have no teacher-forced model loss.

    Identity/constant use observable input length for their own termination;
    oracle explicitly implements the target rule, it is not learned evidence.
    """
    if kind not in {"identity", "constant", "oracle"} or constant_token not in CONTENT_IDS:
        raise ValueError("unknown baseline or invalid constant")
    rows = []
    for record in records:
        content = list(record["content"])
        output = content if kind == "identity" else ([constant_token] * len(content) if kind == "constant" else content[::-1])
        output = (output + [EOS])[:max_new_tokens]
        rows.append({"id": record["id"], "group_id": record["group_id"], "content": content,
                     "target": list(record["target"]), "length": record["length"], "predicted": output,
                     **score_generation(output, record["target"], max_new_tokens=max_new_tokens)})
    return {"metrics": {**summarize_predictions(rows), "teacher_forced_loss": None,
                        "teacher_forced_token_accuracy": None, "max_new_tokens": max_new_tokens,
                        "generation": "algorithmic-oracle" if kind == "oracle" else "task-aware-deterministic-baseline"},
            "predictions": rows,
            "by_length": {str(n): summarize_predictions([r for r in rows if r["length"] == n]) for n in sorted({r["length"] for r in rows})}}


def most_common_training_symbol(records: Sequence[dict]) -> int:
    counts = {token: sum(r["target"].count(token) for r in records) for token in CONTENT_IDS}
    return max(CONTENT_IDS, key=lambda token: (counts[token], -token))


def capture_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state().clone()}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


class RecordBatchStream:
    """Dedicated shuffled order RNG, serializable permutation and data cursor."""
    def __init__(self, records: Sequence[dict], batch_size: int = 64, seed: int = 11):
        if not records or batch_size < 1:
            raise ValueError("batch stream requires records and positive batch size")
        self.records, self.batch_size = list(records), batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(records), generator=self.generator).tolist()
        self.cursor, self.epoch = 0, 0

    def next_records(self) -> list[dict]:
        chosen = []
        while len(chosen) < self.batch_size:
            count = min(self.batch_size - len(chosen), len(self.order) - self.cursor)
            chosen.extend(self.order[self.cursor:self.cursor+count])
            self.cursor += count
            if self.cursor == len(self.order):
                self.epoch += 1
                self.order = torch.randperm(len(self.records), generator=self.generator).tolist()
                self.cursor = 0
        return [self.records[i] for i in chosen]

    def state_dict(self) -> dict:
        return {"record_ids": [r["id"] for r in self.records], "batch_size": self.batch_size,
                "generator": self.generator.get_state().clone(), "order": list(self.order),
                "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        if state["record_ids"] != [r["id"] for r in self.records] or state["batch_size"] != self.batch_size:
            raise ValueError("batch stream data or batch size differs from checkpoint")
        if sorted(state["order"]) != list(range(len(self.records))) or not 0 <= state["cursor"] < len(self.records):
            raise ValueError("invalid saved order/cursor")
        self.generator.set_state(state["generator"])
        self.order, self.cursor, self.epoch = list(state["order"]), state["cursor"], state["epoch"]


def activation_summary(tensor: Tensor) -> dict:
    value = tensor.detach().float()
    return {"rms": float(value.square().mean().sqrt()), "max_abs": float(value.abs().max()),
            "finite": bool(torch.isfinite(value).all()), "shape": list(value.shape)}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_config(model: SmallDecoder) -> dict:
    return asdict(model.config)
