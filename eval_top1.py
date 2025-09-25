import json
import random
import pandas as pd
from datetime import datetime
from app.database.vector_store import VectorStore

# ---------------- config ----------------
GROUNDTRUTH_PATH = "groundtruth.json"
OUT_CSV = "top1_eval_results.csv"
SAMPLE_SIZE =50   # ex: 50 or None pour tout
SEED = 42
# ----------------------------------------

def load_groundtruth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_queries(gt):
    queries = []
    for doc_id, qlist in gt.items():
        for q in qlist:
            if q and q.strip():
                queries.append({"question": q.strip(), "expected_doc_id": doc_id.strip()})
    return queries

def extract_retrieved_id(results):
    """
    results peut être DataFrame (préféré) ou liste de tuples.
    On essaye d'obtenir l'id du doc retourné (et on cherche aussi metadata.doc_id si présent).
    """
    if results is None:
        return None
    # DataFrame case
    if isinstance(results, pd.DataFrame):
        if results.empty:
            return None
        row0 = results.iloc[0]
        # priorité: metadata doc_id si présent, sinon `id`
        # si colonne 'metadata' existe et est un dict, essayer d'extraire 'doc_id'
        if "doc_id" in row0.index:  # si _create_dataframe_from_results_ avait déjà étalé metadata
            candidate = row0.get("doc_id")
            return candidate.strip() if isinstance(candidate, str) else candidate
        # fallback to 'id' column
        if "id" in row0.index:
            candidate = row0.get("id")
            return str(candidate).strip() if pd.notna(candidate) else None
        # else try metadata column if present as dict-like
        if "metadata" in row0.index:
            meta = row0.get("metadata")
            try:
                # meta peut être dict-like
                if isinstance(meta, dict) and "doc_id" in meta:
                    return meta["doc_id"]
            except Exception:
                pass
        return None

    # List / tuple case: assume list of tuples (id, metadata, content, embedding, distance)
    if isinstance(results, (list, tuple)) and len(results) > 0:
        first = results[0]
        # heuristique : if first is tuple-like
        if isinstance(first, (list, tuple)):
            # try positions: 0=id, 1=metadata
            candidate = first[0]
            if candidate is not None:
                return str(candidate).strip()
            # try metadata
            meta = first[1] if len(first) > 1 else None
            if isinstance(meta, dict) and "doc_id" in meta:
                return meta["doc_id"]
    return None

def run_evaluation(vec: VectorStore, queries, sample_size=None, seed=42):
    if sample_size:
        random.Random(seed).shuffle(queries)
        queries = queries[:sample_size]

    records = []
    for i, q in enumerate(queries, start=1):
        question = q["question"]
        expected = q["expected_doc_id"]

        try:
            results = vec.search(question, limit=1)  # return DataFrame per your implementation
        except Exception as e:
            print(f"[{i}/{len(queries)}] ERROR for question: {e}")
            retrieved_id = None
            distance = None
        else:
            retrieved_id = extract_retrieved_id(results)
            # try extract distance if DataFrame
            if isinstance(results, pd.DataFrame) and not results.empty and "distance" in results.columns:
                distance = results.iloc[0].get("distance")
            else:
                distance = None

        correct = (retrieved_id == expected)
        records.append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "expected_doc_id": expected,
            "retrieved_doc_id": retrieved_id,
            "distance": distance,
            "correct": correct
        })
        if i % 10 == 0:
            print(f"[{i}/{len(queries)}] processed - correct so far: {sum(r['correct'] for r in records)}")

    df = pd.DataFrame(records)
    return df

def main():
    gt = load_groundtruth(GROUNDTRUTH_PATH)
    queries = build_queries(gt)
    print(f"Total queries loaded: {len(queries)}")

    if SAMPLE_SIZE:
        sample_size = SAMPLE_SIZE
    else:
        sample_size = len(queries)

    vec = VectorStore()
    df_results = run_evaluation(vec, queries, sample_size, seed=SEED)

    # metrics
    total = len(df_results)
    correct = df_results["correct"].sum()
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTop-1 accuracy: {accuracy:.4f} ({correct}/{total})")

    # save
    df_results.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Results saved to {OUT_CSV}")

if __name__ == "__main__":
    main()
