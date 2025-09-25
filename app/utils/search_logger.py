import pandas as pd

class SearchLogger:
    def __init__(self):
        # This will hold all queries and their results
        self.query_log = pd.DataFrame(columns=["query", "results"])

    def log_search(self, query: str, results: pd.DataFrame):
        """Store a query and its results in the log DataFrame"""
        # Save results as JSON string for compact storage
        row = {
            "query": query,
            "results": results.to_dict(orient="records")
        }
        self.query_log = pd.concat([self.query_log, pd.DataFrame([row])], ignore_index=True)

    def export(self, path="query_results_log.parquet"):
        """Export logs to disk for later evaluation"""
        self.query_log.to_parquet(path, index=False)
        print(f"✅ Saved log to {path}")
