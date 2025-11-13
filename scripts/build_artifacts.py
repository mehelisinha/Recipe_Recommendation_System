import os
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "1_Recipe_csv.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv(DATA_CSV)

# Replace pure empty strings with NA, then drop rows missing key fields
df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

required_cols = {"recipe_title", "ingredients"}
missing_required = required_cols - set(df.columns)
if missing_required:
    raise KeyError(f"Missing required column(s) in CSV: {missing_required}")

df = df.dropna(subset=list(required_cols)).copy()

# Drop duplicates by title + ingredients to avoid near-identical recipes
df = df.drop_duplicates(subset=["recipe_title", "ingredients"]).copy()

# Clean up recipe_title (strip stray quotes, etc.)
df["recipe_title"] = (
    df["recipe_title"]
    .astype(str)
    .str.strip('"\'' )
    .str.replace('"', "", regex=False)
)

# -------------------------------------------------------------------
# Normalization helper
# -------------------------------------------------------------------
def norm(s):
    return str(s).lower().replace("\n", " ").replace("\r", " ").strip()

# Normalized fields
df["title_norm"] = df["recipe_title"].apply(norm)
df["desc_norm"]  = df["description"].apply(norm) if "description" in df.columns else ""
df["ing_norm"]   = df["ingredients"].apply(norm)
df["cat_norm"]   = df["category"].apply(norm) if "category" in df.columns else ""
df["subcat_norm"] = df["subcategory"].apply(norm) if "subcategory" in df.columns else ""

# -------------------------------------------------------------------
# Feature corpus
#   - include category + subcategory
#   - give extra weight to ingredients by repeating them
# -------------------------------------------------------------------
df["text_corpus"] = (
    df["title_norm"]
    + " [cat] " + df["cat_norm"]
    + " [sub] " + df["subcat_norm"]
    + " [ing] " + df["ing_norm"] + " " + df["ing_norm"]   # 2x weight for ingredients
    + " [desc] " + df["desc_norm"]
)

# -------------------------------------------------------------------
# Save metadata for UI
# -------------------------------------------------------------------
meta_cols = [
    "recipe_title", "category", "subcategory",
    "description", "ingredients", "directions",
    "num_ingredients", "num_steps"
]
keep_cols = [c for c in meta_cols if c in df.columns]
df[keep_cols].to_csv(os.path.join(MODELS_DIR, "metadata.csv"), index=False)

# -------------------------------------------------------------------
# Vectorize text_corpus
# -------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
    min_df=2
)
X = vectorizer.fit_transform(df["text_corpus"])

# -------------------------------------------------------------------
# Fit nearest neighbors (cosine)
# -------------------------------------------------------------------
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(X)

# -------------------------------------------------------------------
# Persist artifacts
# -------------------------------------------------------------------
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
sparse.save_npz(os.path.join(MODELS_DIR, "tfidf_matrix.npz"), X)
joblib.dump(nn, os.path.join(MODELS_DIR, "nearest_neighbors_cosine.joblib"))

print("✅ Artifacts written to:", MODELS_DIR)
print("   #recipes used:", len(df))
