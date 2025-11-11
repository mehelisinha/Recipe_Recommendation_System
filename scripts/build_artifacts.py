import os
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(PROJECT_ROOT\data, "1_Recipe_csv.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_CSV)

def norm(s):
    return str(s).lower().replace("\n", " ").replace("\r", " ").strip()

df["title_norm"] = df["recipe_title"].apply(norm)
df["desc_norm"]  = df["description"].apply(norm)
df["ing_norm"]   = df["ingredients"].apply(norm)
df["text_corpus"] = df["title_norm"] + " [ING] " + df["ing_norm"] + " [DESC] " + df["desc_norm"]

# Save metadata for UI
meta_cols = ["recipe_title","category","subcategory","description","ingredients","directions","num_ingredients","num_steps"]
df[meta_cols].to_csv(os.path.join(MODELS_DIR, "metadata.csv"), index=False)

# Vectorize
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=80000,
    min_df=2
)
X = vectorizer.fit_transform(df["text_corpus"])

# Fit nearest neighbors (cosine)
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(X)

# Persist
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
sparse.save_npz(os.path.join(MODELS_DIR, "tfidf_matrix.npz"), X)
joblib.dump(nn, os.path.join(MODELS_DIR, "nearest_neighbors_cosine.joblib"))

print("✅ Artifacts written to:", MODELS_DIR)