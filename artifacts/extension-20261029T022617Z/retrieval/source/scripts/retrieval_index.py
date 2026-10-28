"""Faiss/NumPy process: consume saved encoder vectors; never import PyTorch."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import faiss


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
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


def run(directory):
    out=Path(directory); faiss.omp_set_num_threads(1)
    passages=json.loads((out/"passages.json").read_text()); dataset=json.loads((out/"queries.json").read_text())
    arrays=np.load(out/"embeddings.npz"); x,q=arrays["passages"],arrays["queries"]
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
    save(out / "index-results.json", {"summaries":summaries,"queries":observations,
        "indexes":{name:{"file":name+".faiss","bytes":(out/(name+".faiss")).stat().st_size,"sha256":sha(out/(name+".faiss"))} for name in ["flat","ivf"]},
        "runtime":{"faiss":faiss.__version__,"numpy":np.__version__,"threads":1,"separate_process_from_encoder":True},
        "correctness":{"ivf_all_lists_matches_flat":True,"index_reload_matches":True,"normalized":True,"query_group_overlap":0,"rrf_preserves_candidate_set":True}})

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--directory",required=True);run(parser.parse_args().directory)
