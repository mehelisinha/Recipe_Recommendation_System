import streamlit as st
import pandas as pd
import sys
import os
import base64

# ------------------------------------------------------------
# Path setup so we can import from src/
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommender import RecipeRecommender
from src.utils import clean_user_ingredients, pretty_list

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍝",
    layout="wide",
)

# ------------------------------------------------------------
# Global CSS: remove top padding, style cards, etc.
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Reduce overall top padding */
    .block-container {
        padding-top: 1rem !important;
    }

    /* Reduce sidebar top padding */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    .card {
        background-color: #ffffff;
        padding: 1.0rem 1.2rem;
        border-radius: 0.9rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.0rem;
        border: 1px solid #e5e7eb;
    }

    .badge {
        display: inline-block;
        padding: 0.12rem 0.6rem;
        border-radius: 999px;
        background-color: #f97316;
        color: white;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .small-label {
        font-size: 0.75rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Hero banner CSS injection
# ------------------------------------------------------------
def _inject_hero_css(image_path: str):
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    hero_css = f"""
        <style>
        .hero-wrapper {{
            position: relative;
            height: 220px;  /* compact banner */
            border-radius: 0.8rem;
            overflow: hidden;
            margin-top: 0.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
            background-image:
                linear-gradient(
                    to right,
                    rgba(15,23,42,0.85),
                    rgba(15,23,42,0.55),
                    rgba(15,23,42,0.25)
                ),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        .hero-content {{
            padding: 1.2rem 2rem;
            color: #f9fafb;
        }}
        .hero-title {{
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }}
        .hero-subtitle {{
            font-size: 0.92rem;
            max-width: 45rem;
            color: #e5e7eb;
        }}
        .hero-bullets {{
            margin-top: 0.4rem;
            font-size: 0.85rem;
            color: #e5e7eb;
        }}
        </style>
    """
    st.markdown(hero_css, unsafe_allow_html=True)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
@st.cache_resource
def load_system():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    rec = RecipeRecommender(model_dir)
    return rec

rec = load_system()

# ------------------------------------------------------------
# Hero banner
# ------------------------------------------------------------
HERO_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "hero_recipes.jpg")
_inject_hero_css(HERO_IMAGE_PATH)

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-content">
            <div class="hero-title">Content-Based Recipe Recommender</div>
            <div class="hero-subtitle">
                Find recipes that match your ingredients, cravings, and style — powered by TF-IDF and cosine similarity.
            </div>
            <div class="hero-bullets">
                • <b>By ingredients</b>: type what you have at home and get matching recipes.<br/>
                • <b>Similar to a recipe</b>: paste a recipe title to discover look-alikes.<br/>
                • Sorted from best to worst match based on cosine similarity.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ------------------------------------------------------------
# Sidebar – search controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("Search")
    mode = st.radio("Mode", ["By ingredients / free text", "Similar to a recipe"])

    if mode == "By ingredients / free text":
        ing = st.text_area(
            "Ingredients (comma-separated)",
            placeholder="tomato, onion, garlic, basil",
            height=110,
        )
        free = st.text_input(
            "Optional: style or description",
            placeholder="quick dinner, vegan, spicy",
        )
        topk = st.slider("How many results?", min_value=5, max_value=30, value=10, step=1)
        category = st.text_input("Filter: Category contains", "")
        subcategory = st.text_input("Filter: Subcategory contains", "")
        run = st.button("Search", type="primary")
        title = ""
    else:
        ing = None
        free = None
        category = ""
        subcategory = ""
        title = st.text_input("Recipe title", placeholder="e.g. Churros II")
        topk = st.slider("How many results?", min_value=5, max_value=30, value=10, step=1)
        run = st.button("Find similar", type="primary")

# ------------------------------------------------------------
# Helper: format for display
# ------------------------------------------------------------
def _prepare_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(lambda v: pretty_list(v, bullet=True))
    if "directions" in df.columns:
        df["directions"] = df["directions"].apply(lambda v: pretty_list(v, bullet=True))
    if "similarity" in df.columns:
        df["similarity"] = df["similarity"].astype(float).clip(0, 1)
    return df

# ------------------------------------------------------------
# Card rendering (NO assistant, just recipe cards)
# ------------------------------------------------------------
def render_cards(df: pd.DataFrame):
    """
    Show clean cards with title/category/description and full recipe details.
    No chatbot / assistant.
    """
    if df is None or df.empty:
        st.info("No recipes to display.")
        return

    df = _prepare_for_display(df)
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    for idx, row in df.iterrows():
        title = row.get("recipe_title", "Untitled recipe")
        category = row.get("category", "") or ""
        subcat = row.get("subcategory", "") or ""
        desc = row.get("description", "") or ""
        ingredients = row.get("ingredients", "") or ""
        directions = row.get("directions", "") or ""
        n_ing = row.get("num_ingredients", None)
        n_steps = row.get("num_steps", None)
        rank = int(row.get("rank", idx + 1))

        # Card header
        st.markdown(
            f"""
            <div class="card">
                <div>
                    <div class="badge">#{rank} Recommendation</div>
                    <h3 style="margin:0.35rem 0 0.15rem 0;">{title}</h3>
                    <div style="color:#6b7280;">
                        {category if category else "Uncategorized"}
                        {" • " + subcat if subcat else ""}
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # metadata
        meta = []
        if n_ing is not None and not pd.isna(n_ing):
            meta.append(f"{int(n_ing)} ingredients")
        if n_steps is not None and not pd.isna(n_steps):
            meta.append(f"{int(n_steps)} steps")

        if meta:
            st.markdown(
                f'<div style="font-size:0.80rem;color:#4b5563;margin-top:0.25rem;">{" • ".join(meta)}</div>',
                unsafe_allow_html=True,
            )

        if isinstance(desc, str) and desc.strip():
            st.markdown(
                f'<div style="font-size:0.85rem;color:#4b5563;margin-top:0.35rem;">{desc}</div>',
                unsafe_allow_html=True,
            )

        # --- Full recipe expander (ingredients + directions) ---
        with st.expander("View full recipe"):
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Ingredients**")
                if isinstance(ingredients, str) and ingredients.strip():
                    st.markdown(ingredients.replace("\n", "<br/>"), unsafe_allow_html=True)
                else:
                    st.write("No ingredients available.")
            with colB:
                st.markdown("**Directions**")
                if isinstance(directions, str) and directions.strip():
                    st.markdown(directions.replace("\n", "<br/>"), unsafe_allow_html=True)
                else:
                    st.write("No directions available.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Optional: table view
    with st.expander("View all results as table"):
        st.dataframe(df.drop(columns=["similarity"], errors="ignore"), use_container_width=True)

# ------------------------------------------------------------
# Main logic – run search & show results
# ------------------------------------------------------------
if run:
    with st.spinner("Searching recipes..."):
        if mode == "By ingredients / free text":
            q = ""
            if ing:
                q += "[ING] " + clean_user_ingredients(ing)
            if free:
                q += " [DESC] " + free

            if not q.strip():
                st.warning("Please provide ingredients and/or a short description.")
            else:
                filters = {}
                if category:
                    filters["category"] = category
                if subcategory:
                    filters["subcategory"] = subcategory

                out = rec.search(q, top_k=topk, filters=filters)

                if out.empty:
                    st.info("No matching recipes found. Try fewer filters or different keywords.")
                else:
                    st.subheader("Recommended recipes")
                    render_cards(out)

        else:  # Similar to a recipe
            if not title.strip():
                st.warning("Enter a recipe title to find similar dishes.")
            else:
                out = rec.recommend_like(title, top_k=topk)

                if out.empty:
                    st.info("No similar recipes found. Try another recipe title.")
                else:
                    st.subheader(f"Recipes similar to: _{title}_")
                    render_cards(out)

st.caption(
    "Model: TF-IDF (1–2 grams) over title + category + subcategory + ingredients + description; "
    "similarity: cosine; results ranked by similarity (scores hidden for a cleaner UX)."
)
