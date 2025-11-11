import streamlit as st
import pandas as pd
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommender import RecipeRecommender
from src.utils import clean_user_ingredients


st.set_page_config(page_title="Recipe Recommender", layout="wide")

@st.cache_resource
def load_system():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    rec = RecipeRecommender(model_dir)
    return rec

rec = load_system()

st.title("Content-Based Recipe Recommender")
st.markdown("Find recipes by ingredients or discover similar dishes.")

with st.sidebar:
    st.header("Search")
    mode = st.radio("Mode", ["By ingredients / free text", "Similar to a recipe"])

    if mode == "By ingredients / free text":
        ing = st.text_area("Ingredients (comma-separated)", placeholder="tomato, onion, garlic, basil", height=100)
        free = st.text_input("Optional: style or description", placeholder="quick dinner, vegan, spicy")
        topk = st.slider("How many results?", min_value=5, max_value=50, value=15, step=5)
        category = st.text_input("Filter: Category contains", "")
        subcategory = st.text_input("Filter: Subcategory contains", "")
        run = st.button("Search")
    else:
        title = st.text_input("Recipe title", "")
        topk = st.slider("How many results?", min_value=5, max_value=50, value=15, step=5)
        run = st.button("Find similar")

tab1, tab2 = st.tabs(["Results", "Recipe browser"])

if run:
    with st.spinner("Searching..."):
        if mode == "By ingredients / free text":
            q = ""
            if ing:
                q += "[ING] " + clean_user_ingredients(ing)
            if free:
                q += " [DESC] " + free
            if not q.strip():
                st.warning("Please provide ingredients and/or a description.")
            else:
                filters = {}
                if 'category' in rec.meta.columns and category:
                    filters['category'] = category
                if 'subcategory' in rec.meta.columns and subcategory:
                    filters['subcategory'] = subcategory
                out = rec.search(q, top_k=topk, filters=filters)
                with tab1:
                    st.dataframe(out[["recipe_title","category","subcategory",
                                      "num_ingredients","num_steps","similarity",
                                      "ingredients","directions","description"]])
        else:
            if not title.strip():
                st.warning("Enter a recipe title.")
            else:
                out = rec.recommend_like(title, top_k=topk)
                with tab1:
                    st.dataframe(out[["recipe_title","category","subcategory",
                                      "num_ingredients","num_steps","similarity",
                                      "ingredients","directions","description"]])

with tab2:
    st.write("Sample of the catalog:")
    st.dataframe(rec.meta.head(200))

st.caption("Model: TF-IDF (1–2 grams) over title + ingredients + description; similarity: cosine; index: brute-force NearestNeighbors.")