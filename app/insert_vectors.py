import json
import pandas as pd
from app.database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from datetime import datetime


# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/Rdataset.csv", sep=";")

def prepare_record(row):
    content = row["content"]
    embedding = vec.get_embedding(content)

    # Parse metadata if it's a string
    metadata = row["metadata"]
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {"raw": metadata}

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": metadata,
            "contents": content,
            "embedding": embedding,
        }
    )
# Apply transformation
records_df = df.apply(prepare_record, axis=1)



# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)

print(f"✅ {len(records_df)} documents inserted into the vector store")
