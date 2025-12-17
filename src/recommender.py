import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Any
from src.utils import clean_text_list, clean_user_query_ingredients, normalize_ingredient

class RecipeRecommender:
    def __init__(self, model_dir: str):
        """
        Initialize the recommender system by loading models and data.
        """
        self.model_dir = model_dir
        
        # Load artifacts
        # We assume these exist. If not, the system should probably run build_artifacts.py
        # But for now we just load them.
        self.vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.joblib")
        self.X = sparse.load_npz(f"{model_dir}/tfidf_matrix.npz")
        self.nn = joblib.load(f"{model_dir}/nearest_neighbors_cosine.joblib")
        
        # Load metadata
        self.meta = pd.read_csv(f"{model_dir}/metadata.csv")
        self.meta["id"] = np.arange(len(self.meta))
        
        # Optimization: cache parsed ingredients to avoid expensive ast.literal_eval on every startup
        cache_path = f"{model_dir}/parsed_ingredients.joblib"
        if os.path.exists(cache_path):
            self.parsed_ingredients = joblib.load(cache_path)
            # Basic sanity check
            if len(self.parsed_ingredients) != len(self.meta):
                self._parse_and_cache_ingredients(cache_path)
        else:
            self._parse_and_cache_ingredients(cache_path)

    def _parse_and_cache_ingredients(self, cache_path: str):
        """
        Parse stringified ingredient lists into sets of normalized strings.
        Save to disk for faster future loads.
        """
        if "ingredients" in self.meta.columns:
            # Optimize: straightforward list comprehension is often faster than vectorization for string ops
            # clean_text_list can be slow if it uses ast.literal_eval on 60k rows.
            # We accept the startup cost once, then cache.
            parsed = [
                set(normalize_ingredient(i) for i in clean_text_list(row))
                for row in self.meta["ingredients"]
            ]
            self.parsed_ingredients = parsed
        else:
            self.parsed_ingredients = [set() for _ in range(len(self.meta))]
            
        joblib.dump(self.parsed_ingredients, cache_path)

    def _compute_overlap_score(self, candidate_indices: np.ndarray, user_ingredients: set) -> np.ndarray:
        """
        Compute a score (0.0 to 1.0) based on how many user ingredients cover the recipe's ingredients.
        We care about RECALL (how much of the user's input is used) and PRECISION (does the recipe require extra stuff?).
        """
        if not user_ingredients:
            return np.zeros(len(candidate_indices))
            
        scores = []
        for idx in candidate_indices:
            idx = int(idx)
            # access pre-parsed list
            recipe_ingredients = self.parsed_ingredients[idx]
            
            if not recipe_ingredients:
                scores.append(0.0)
                continue
            
            # IMPROVED LOGIC: Substring match
            # If user says "basil", and recipe has "1 cup chopped basil", it should match.
            # We iterate: for each user ing, is it present in ANY of the recipe ingredients?
            matches = 0
            for u_ing in user_ingredients:
                # Check if u_ing matches any recipe ingredient (substring)
                if any(u_ing in r_ing for r_ing in recipe_ingredients):
                    matches += 1
            
            # Score: fraction of user ingredients found in the recipe
            score = matches / len(user_ingredients) if user_ingredients else 0.0
            scores.append(score)
            
        return np.array(scores)

    def search(self, query: str, top_k: int = 20, filters: Dict[str, Any] = None, match_count: bool = False) -> pd.DataFrame:
        """
        Hybrid search: TF-IDF + Ingredient Overlap
        match_count: If True, only return recipes with strictly the same number of ingredients as query.
        """
        # 1. Broad retrieval via TF-IDF
        vec = self.vectorizer.transform([query])
        # Get more candidates than top_k to allow re-ranking
        # If strict matching is on, we need A LOT of candidates because exact match is rare
        n_candidates = 2000 if match_count else 200
        n_candidates = min(n_candidates, self.X.shape[0])
        
        dists, inds = self.nn.kneighbors(vec, n_neighbors=n_candidates)
        
        inds = inds[0]
        dists = dists[0]
        tfidf_sim = 1.0 - dists
        
        # 2. Ingredient overlap re-ranking
        user_ingredients = clean_user_query_ingredients(query)
        overlap_scores = self._compute_overlap_score(inds, user_ingredients)
        
        # 3. Hybrid Score
        if len(user_ingredients) > 1:
            alpha = 0.6  # Weight for overlap
            final_scores = (1 - alpha) * tfidf_sim + alpha * overlap_scores
        else:
            final_scores = tfidf_sim
            
        # Create result dataframe
        results = self.meta.iloc[inds].copy()
        results["similarity"] = final_scores
        results["tfidf_score"] = tfidf_sim
        results["overlap_score"] = overlap_scores
        
        # --- STRICT COUNT MATCHING ---
        if match_count and len(user_ingredients) > 0:
            user_count = len(user_ingredients)
            # Filter results where cached ingredient count == user_count
            # We need to look up counts.
            
            # Optimization: Pre-calculate counts for candidates?
            # Or just filter the dataframe since we have cache.
            
            # Map index
            strict_mask = []
            for idx in inds: # iter over original indices
                 # parsed_ingredients corresponds to self.meta index
                 r_count = len(self.parsed_ingredients[idx])
                 strict_mask.append(r_count == user_count)
            
            results = results[strict_mask]
        
        # 4. Filters
        if filters:
            for col, val in filters.items():
                if val and col in results.columns:
                    mask = results[col].astype(str).str.contains(str(val), case=False, na=False)
                    results = results[mask]
        
        # 5. Sort and return
        results = results.sort_values(by="similarity", ascending=False).head(top_k)
        return results.reset_index(drop=True)

    def recommend_like(self, recipe_title: str, top_k: int = 20) -> pd.DataFrame:
        """
        Recommend recipes similar to a given title.
        """
        # Case insensitive match
        mask = self.meta["recipe_title"].str.strip().str.lower() == str(recipe_title).strip().lower()
        if not mask.any():
            return pd.DataFrame()
            
        idx = self.meta.index[mask][0]
        
        vec = self.X[idx]
        dists, inds = self.nn.kneighbors(vec, n_neighbors=min(top_k+1, self.X.shape[0]))
        
        inds = inds[0][1:]
        dists = dists[0][1:]
        
        results = self.meta.iloc[inds].copy()
        results["similarity"] = 1.0 - dists
        return results.reset_index(drop=True)
