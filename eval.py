# Setup
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


QUERIED_CSV_PATH = "all_outputs.csv"
VALIDATION_DATA_PATH = "validation.csv"

VALIDATION_COLS = ["B.Code", "A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib", "D.Notes"]
validation_df = pd.read_csv(VALIDATION_DATA_PATH, header=None, names=VALIDATION_COLS)
queried_df = pd.read_csv(QUERIED_CSV_PATH)

valItems = validation_df["D.Item"].values.tolist()[:14]
queryItems = queried_df["D.Item"].values.tolist()[:14]

model = SentenceTransformer("all-MiniLM-L6-v2")

#recobo/agri-sentence-transformer
vectors = model.encode(valItems)

# Function to get top N similar items
def top_similar(query, items, vectors, topn=3):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, vectors)[0]
    top_idx = np.argsort(-sims)[:topn]
    return [(items[i], sims[i]) for i in top_idx]

# Example: most similar to "red apple"
results = []
for i in range(len(queryItems)):
    results.append((top_similar(queryItems[i], valItems, vectors), valItems[i]))


for i in range(len(results)):
    print(results[i])

s = model.encode("Maize Ground")
v = model.encode("Yellow maize meal")

#print(cosine_similarity([s, v]))
