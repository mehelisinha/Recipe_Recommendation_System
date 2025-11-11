import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse

class RecipeRecommender:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.joblib")
        self.X = sparse.load_npz(f"{model_dir}/tfidf_matrix.npz")
        self.nn = joblib.load(f"{model_dir}/nearest_neighbors_cosine.joblib")
        self.meta = pd.read_csv(f"{model_dir}/metadata.csv")
        self.meta["id"] = np.arange(len(self.meta))

    def search(self, query: str, top_k: int = 20, filters=None):
        vec = self.vectorizer.transform([query])
        dists, inds = self.nn.kneighbors(vec, n_neighbors=min(top_k, self.X.shape[0]))
        inds = inds[0]; dists = dists[0]
        results = self.meta.iloc[inds].copy()
        results["similarity"] = 1.0 - dists
        if filters:
            for col, val in filters.items():
                if val and col in results.columns:
                    results = results[results[col].astype(str)
                                      .str.contains(str(val), case=False, na=False)]
        return results.reset_index(drop=True)

    def recommend_like(self, recipe_title: str, top_k: int = 20):
        mask = self.meta["recipe_title"].str.lower() == str(recipe_title).lower()
        if mask.any():
            idx = self.meta.index[mask][0]
        else:
            idx = 0
        vec = self.X[idx]
        dists, inds = self.nn.kneighbors(vec, n_neighbors=min(top_k+1, self.X.shape[0]))
        inds = inds[0][1:]; dists = dists[0][1:]
        results = self.meta.iloc[inds].copy()
        results["similarity"] = 1.0 - dists
        return results.reset_index(drop=True)