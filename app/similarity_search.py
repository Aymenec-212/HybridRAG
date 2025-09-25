from datetime import datetime
from app.database.vector_store import VectorStore
from app.services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# question
# --------------------------------------------------------------

relevant_question = "Entrez vos questions ici "
results = vec.search(relevant_question, limit=5)

# Save to CSV
results.to_csv("search_results_.csv", index=False)
# --------------------------------------------------------------
# response
# --------------------------------------------------------------
response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")


# --------------------------------------------------------------
# Time-based filtering
# --------------------------------------------------------------


time_range = (datetime(1979, 4, 9), datetime(2010, 4, 13))
results = vec.search(relevant_question, limit=3, time_range=time_range)

time_range = (datetime(1975, 2, 9), datetime(1980, 5, 14))
results = vec.search(relevant_question, limit=3, time_range=time_range)