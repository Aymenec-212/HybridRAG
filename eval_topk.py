#!/usr/bin/env python3
"""
eval_topk.py

Évaluation Top-K (Hit@K) et MRR pour votre système de retrieval,
en utilisant groundtruth.json (format: { "doc_id": ["Q1","Q2","Q3"], ... }).

Usage:
    python eval_topk.py
"""

from __future__ import annotations
import json
import random
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.database.vector_store import VectorStore

# ---------------- config ----------------
GROUNDTRUTH_PATH = "groundtruth1.json"
OUT_CSV = "topk_eval_results.csv"
KS = [1, 3, 5]        # valeurs de k à évaluer (peuvent être modifiées)
SAMPLE_SIZE = 50    # mettre un entier pour échantillonner (ex: 50), ou None pour tout
SEED = 42
# ----------------------------------------

def load_groundtruth(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_queries(gt: Dict[str, List[str]]) -> List[Dict[str, str]]:
    queries = []
    for doc_id, qlist in gt.items():
        for q in qlist:
            if q and str(q).strip():
                queries.append({"question": str(q).strip(), "expected_doc_id": str(doc_id).strip()})
    return queries

def extract_ids_and_distances(results: Any, top_k: int) -> (List[Optional[str]], List[Optional[float]]):
    """
    Retourne (ids_list, distances_list) extraits des résultats retournés par vec.search.
    Gère DataFrame (préféré) ou liste/tuple de tuples.
    """
    ids: List[Optional[str]] = []
    dists: List[Optional[float]] = []

    if results is None:
        return ids, dists

    # DataFrame case
    if isinstance(results, pd.DataFrame):
        if results.empty:
            return ids, dists
        df = results.head(top_k)
        for _, row in df.iterrows():
            # Priority: column 'doc_id' (étalée), then 'id', then metadata dict
            candidate = None
            if "doc_id" in row.index and pd.notna(row.get("doc_id")):
                candidate = row.get("doc_id")
            elif "id" in row.index and pd.notna(row.get("id")):
                candidate = row.get("id")
            elif "metadata" in row.index:
                meta = row.get("metadata")
                if isinstance(meta, dict):
                    # common keys
                    for key in ("doc_id", "docId", "id"):
                        if key in meta and meta[key] is not None:
                            candidate = meta[key]
                            break
            ids.append(str(candidate).strip() if candidate is not None else None)

            # distance if present
            if "distance" in row.index:
                try:
                    dists.append(float(row.get("distance")))
                except Exception:
                    dists.append(None)
            else:
                dists.append(None)
        return ids, dists

    # List / tuple case (assume list of results)
    if isinstance(results, (list, tuple)):
        for item in list(results)[:top_k]:
            cand = None
            dist = None
            if isinstance(item, (list, tuple)):
                # common layout: (id, metadata, content, embedding, distance)
                if len(item) > 0 and item[0] is not None:
                    cand = item[0]
                if len(item) > 1 and isinstance(item[1], dict):
                    for key in ("doc_id", "docId", "id"):
                        if key in item[1] and item[1][key] is not None:
                            cand = item[1][key]
                            break
                # distance often last
                if len(item) >= 5:
                    try:
                        dist = float(item[-1])
                    except Exception:
                        dist = None
            else:
                # unknown object
                cand = str(item)
            ids.append(str(cand).strip() if cand is not None else None)
            dists.append(dist)
        return ids, dists

    # fallback
    return ids, dists

def run_evaluation(vec: VectorStore, queries: List[Dict[str, str]], ks: List[int], sample_size: Optional[int]=None, seed: int=42) -> pd.DataFrame:
    if sample_size:
        rnd = random.Random(seed)
        rnd.shuffle(queries)
        queries = queries[:sample_size]

    max_k = max(ks)
    records = []

    for i, q in enumerate(queries, start=1):
        question = q["question"]
        expected = q["expected_doc_id"]

        try:
            results = vec.search(question, limit=max_k)
        except Exception as e:
            print(f"[{i}/{len(queries)}] ERROR for question: {e}")
            ids, dists = [], []
        else:
            ids, dists = extract_ids_and_distances(results, max_k)

        # compute hit@k for each k and rank info
        hits = {}
        for k in ks:
            topk_ids = ids[:k]
            hits[f"hit@{k}"] = (expected in topk_ids) if topk_ids else False

        # reciprocal rank
        rr = 0.0
        rank = None
        if ids:
            try:
                idx = ids.index(expected)
                rank = idx + 1
                rr = 1.0 / rank
            except ValueError:
                rr = 0.0
                rank = None

        records.append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "expected_doc_id": expected,
            "retrieved_ids": ids,
            "retrieved_distances": dists,
            "rank_of_expected": rank,
            "reciprocal_rank": rr,
            **{f"hit@{k}": hits[f"hit@{k}"] for k in ks}
        })

        if i % 10 == 0 or i == len(queries):
            correct_so_far = sum(1 for r in records if r["hit@{0}".format(ks[0])])
            print(f"[{i}/{len(queries)}] processed (last query expected={expected})")

    df = pd.DataFrame(records)
    return df

def compute_metrics(df: pd.DataFrame, ks: List[int]) -> Dict[str, Any]:
    metrics = {}
    total = len(df)
    for k in ks:
        metrics[f"Hit@{k}"] = df[f"hit@{k}"].mean() if f"hit@{k}" in df.columns else 0.0
    metrics["MRR"] = df["reciprocal_rank"].mean() if "reciprocal_rank" in df.columns else 0.0
    # average rank for hits (ignore None)
    ranks = df["rank_of_expected"].dropna().astype(float)
    metrics["avg_rank_of_hits"] = ranks.mean() if not ranks.empty else None
    return metrics

def main():
    gt = load_groundtruth(GROUNDTRUTH_PATH)
    queries = build_queries(gt)
    print(f"Total queries loaded: {len(queries)}")

    vec = VectorStore()
    df_results = run_evaluation(vec, queries, ks=KS, sample_size=SAMPLE_SIZE, seed=SEED)

    # compute and print metrics
    metrics = compute_metrics(df_results, KS)
    print("\n=== Summary metrics ===")
    for k in KS:
        print(f"Hit@{k}: {metrics[f'Hit@{k}']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
    if metrics["avg_rank_of_hits"] is not None:
        print(f"Average rank for hits: {metrics['avg_rank_of_hits']:.2f}")

    # save detailed results
    df_results.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nDetailed results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
