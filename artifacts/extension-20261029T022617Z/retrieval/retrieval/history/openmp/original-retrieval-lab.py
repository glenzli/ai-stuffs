"""Local Chinese retrieval: real paragraphs, pinned encoder, BM25, Faiss and RRF."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys
import platform
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
FROZEN = {"model_id": "BAAI/bge-small-zh-v1.5", "revision": "7999e1d3359715c523056ef9478215996d62a620", "max_length": 512,
          "pooling": "last_hidden_state[:,0] (CLS), then L2 normalize", "device": "cpu", "dtype": "float32", "attention": "eager", "threads": 1,
          "passage_batch": 16, "query_batch": 8, "query_instruction": QUERY_PREFIX, "document_instruction": "none",
          "bm25": {"k1": 1.2, "b": .75, "tokenizer": "lowercase Latin words/numbers and individual CJK characters; no learned segmenter"},
          "ivf": {"nlist": 16, "nprobe": [1, 4, 16], "training_seed": 20261029}, "k": [1, 3, 5, 10], "stored_top_k": 10,
          "reranking": "RRF(c=60) of Flat rank and global BM25 rank, restricted to the same Flat top-10 candidates; no cross-encoder",
          "selection": "all parameters and questions frozen before ranking; no validation or test parameter search"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def lexical_tokens(text): return re.findall(r"[a-z]+(?:_[a-z]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]", text.lower())


def bm25_scores(passages, queries, k1=1.2, b=.75):
    documents = [Counter(lexical_tokens(row["text"])) for row in passages]; lengths = np.array([sum(row.values()) for row in documents], dtype=np.float64)
    avg = float(lengths.mean()); df = Counter(term for row in documents for term in row)
    output = np.zeros((len(queries), len(documents)), dtype=np.float64)
    for i, query in enumerate(queries):
        for term, qtf in Counter(lexical_tokens(query["query"])).items():
            idf = math.log(1 + (len(documents) - df[term] + .5) / (df[term] + .5))
            for j, row in enumerate(documents):
                tf = row[term]
                if tf: output[i, j] += qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[j] / avg))
    return output


def score_ranking(ids, gold, k=10):
    first = next((i + 1 for i, item in enumerate(ids[:k]) if item in gold), None)
    return {"hit_at_1": int(first is not None and first <= 1), "hit_at_3": int(first is not None and first <= 3),
            "hit_at_5": int(first is not None and first <= 5), "hit_at_10": int(first is not None), "mrr_at_10": 0. if first is None else 1 / first}


def run(out_dir, corpus_file, query_file, model_directory):
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); faiss.omp_set_num_threads(1)
    corpus_file, query_file, model_directory = Path(corpus_file), Path(query_file), Path(model_directory)
    corpus, dataset = json.loads(corpus_file.read_text()), json.loads(query_file.read_text())
    if sha(corpus_file) != dataset["corpus_sha256"]: raise ValueError("Query labels and corpus differ")
    groups = {split: {q["group"] for q in dataset["queries"] if q["split"] == split} for split in ["validation", "test"]}
    assert not groups["validation"] & groups["test"]
    passages = [{"id": p["id"], "source_document": d["id"], "paragraph": i, "text": p["text"]} for d in corpus["documents"] for i, p in enumerate(d["paragraphs"])]
    by_id = {p["id"]: p for p in passages}; assert len(by_id) == len(passages)
    for q in dataset["queries"]:
        for gold in q["gold"]:
            assert gold["evidence"] in by_id[gold["passage_id"]]["text"] and gold["source_document"] == by_id[gold["passage_id"]]["source_document"]
    identity = json.loads((model_directory / "model-manifest.json").read_text())
    if identity["model_id"] != FROZEN["model_id"] or identity["revision"] != FROZEN["revision"]: raise ValueError("Unpinned encoder")
    for entry in identity["files"]:
        if sha(model_directory / entry["file"]) != entry["sha256"]: raise ValueError("Changed encoder bytes")
    runtime = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "threads": 1,
               **{name: importlib.metadata.version(name) for name in ["torch", "numpy", "transformers", "tokenizers", "safetensors", "huggingface-hub", "faiss-cpu"]}}
    save(out / "frozen-config.json", {**FROZEN, "frozen_at": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
                                     "corpus_sha256": sha(corpus_file), "queries_sha256": sha(query_file), "model_identity": identity})
    shutil.copy(corpus_file, out / "corpus.json"); shutil.copy(query_file, out / "queries.json"); save(out / "passages.json", passages)
    (out / "model").mkdir()
    for name in [r["file"] for r in identity["files"]] + ["model-manifest.json"]: shutil.copy(model_directory / name, out / "model" / name)
    tokenizer = AutoTokenizer.from_pretrained(out / "model", local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(out / "model", local_files_only=True, trust_remote_code=False, attn_implementation="eager").eval()
    def encode(texts, batch_size):
        arrays, details = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                part = texts[start:start + batch_size]; raw = tokenizer(part, truncation=False, add_special_tokens=True)["input_ids"]
                batch = tokenizer(part, padding=True, truncation=True, max_length=512, return_tensors="pt")
                hidden = model(**batch).last_hidden_state[:, 0]; arrays.append(F_normalize(hidden).numpy().copy())
                for original, ids, mask in zip(raw, batch["input_ids"], batch["attention_mask"]):
                    details.append({"raw_tokens": len(original), "encoded_tokens": int(mask.sum()), "truncated": len(original) > 512, "input_ids": ids[mask.bool()].tolist()})
                print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32), details
    x, passage_encoding = encode([r["text"] for r in passages], 16)
    q, query_encoding = encode([QUERY_PREFIX + r["query"] for r in dataset["queries"]], 8)
    np.savez(out / "embeddings.npz", passages=x, queries=q)
    save(out / "encoding.json", {"passages": passage_encoding, "queries": query_encoding})
    lexical = bm25_scores(passages, dataset["queries"]); np.save(out / "bm25-scores.npy", lexical)
    flat = faiss.IndexFlatIP(x.shape[1]); flat.add(x); scores, ids = flat.search(q, 10); faiss.write_index(flat, str(out / "flat.faiss"))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(x.shape[1]), x.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    ivf.cp.seed = 20261029; ivf.train(x); ivf.add(x); faiss.write_index(ivf, str(out / "ivf.faiss"))
    assert np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-6)
    index_results = {}
    for nprobe in [1, 4, 16]:
        ivf.nprobe = nprobe; value, indexes = ivf.search(q, 10); index_results[f"ivf-{nprobe}"] = (value, indexes)
    assert all(set(a) == set(b) for a, b in zip(ids, index_results["ivf-16"][1]))
    assert np.array_equal(faiss.read_index(str(out / "flat.faiss")).search(q, 10)[1], ids)
    names = [p["id"] for p in passages]; observations = []
    for i, query in enumerate(dataset["queries"]):
        bm_order = sorted(range(len(passages)), key=lambda j: (-float(lexical[i, j]), names[j])); bm_rank = {j: rank + 1 for rank, j in enumerate(bm_order)}
        rrf = [(1 / (60 + rank + 1) + 1 / (60 + bm_rank[int(j)]), int(j)) for rank, j in enumerate(ids[i])]
        rrf.sort(key=lambda row: (-row[0], names[row[1]]))
        rankings = {"bm25": (lexical[i, bm_order[:10]].tolist(), bm_order[:10]), "flat": (scores[i].tolist(), ids[i].tolist()),
                    **{key: (values[i].tolist(), indexes[i].tolist()) for key, (values, indexes) in index_results.items()},
                    "rrf-rerank": ([r[0] for r in rrf], [r[1] for r in rrf])}
        gold = {r["passage_id"] for r in query["gold"]}; methods = {}
        for name, (values, indexes) in rankings.items():
            records = [{"passage_id": names[j], "score": value, "designated_evidence": names[j] in gold} for value, j in zip(values, indexes) if j >= 0]
            methods[name] = {"results": records, "metrics": score_ranking([r["passage_id"] for r in records], gold),
                             "flat_neighbor_recall_at_10": len(set(indexes) & set(ids[i])) / 10 if name.startswith("ivf-") else None}
        assert set(r["passage_id"] for r in methods["rrf-rerank"]["results"]) == set(r["passage_id"] for r in methods["flat"]["results"])
        observations.append({**query, "methods": methods})
    summaries = []
    for split in ["validation", "test"]:
        for kind in ["all", "keyword", "paraphrase"]:
            selected = [r for r in observations if r["split"] == split and (kind == "all" or r["kind"] == kind)]
            for method in observations[0]["methods"]:
                summaries.append({"split": split, "slice": kind, "method": method, "questions": len(selected),
                                  "metrics": {key: sum(r["methods"][method]["metrics"][key] for r in selected) / len(selected) for key in selected[0]["methods"][method]["metrics"]},
                                  "flat_neighbor_recall_at_10": sum(r["methods"][method]["flat_neighbor_recall_at_10"] for r in selected) / len(selected) if method.startswith("ivf-") else None})
    report = {"schema": "wiki-retrieval-v1", "status": "passed", "frozen": FROZEN, "runtime": runtime, "model": identity,
              "dataset": {"passages": len(passages), "documents": len(corpus["documents"]), "questions": len(observations), "validation_groups": sorted(groups["validation"]), "test_groups": sorted(groups["test"]),
                          "corpus_sha256": sha(corpus_file), "queries_sha256": sha(query_file), "annotation_policy": dataset["annotation_policy"],
                          "truncated_passages": sum(r["truncated"] for r in passage_encoding), "truncated_queries": sum(r["truncated"] for r in query_encoding)},
              "summaries": summaries, "queries": observations, "passages": passages,
              "indexes": {name: {"file": name + ".faiss", "bytes": (out / (name + ".faiss")).stat().st_size, "sha256": sha(out / (name + ".faiss"))} for name in ["flat", "ivf"]},
              "correctness": {"ivf_all_lists_matches_flat": True, "index_reload_matches": True, "normalized": True, "query_group_overlap": 0, "rrf_preserves_candidate_set": True},
              "limits": ["repository-authored questions and designated evidence, not exhaustive or blinded relevance labels; no precision claim", "document corpus is the searchable index, not encoder fine-tuning data; pretrained-model contamination not audited", "RRF is an actual deterministic lexical/vector rank fusion, not a neural cross-encoder", "no model training, remote inference or production-latency benchmark; truncation and every retained failure reported"]}
    save(out / "report.json", report); return report


def F_normalize(value): return torch.nn.functional.normalize(value, p=2, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["out-dir", "corpus", "queries", "model-directory"]: parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); run(args.out_dir, args.corpus, args.queries, args.model_directory)
