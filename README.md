# Content-Based Recipe Recommender 🍲

A content-based recipe recommendation system that suggests recipes based on ingredients you have in your pantry or by finding similarities to your favorite dishes. 

**Enhanced with a "Premium" Streamlit UI and a Hybrid Recommendation Engine.**

![Hero](app/assets/hero_recipes.jpg)

## 🚀 Features

### 1. Hybrid Recommendation Engine
The core logic combines two powerful techniques to give you the best results:
- **TF-IDF & Cosine Similarity**: Analyzes the "vibe" of a recipe (title, category, description) to find broadly similar matches.
- **Ingredient Overlap Scoring**: A precision layer that parses your input ingredients and calculates exactly how much of a recipe matches your pantry.
    - *Example*: If you type "chicken, lemon", it prioritizes recipes that actually contain both.

### 2. Search Modes
- **By Ingredients (Precise)**: Enter a list of ingredients (e.g., `tomato, basil, mozzarella`).
- **Similar to a Recipe**: Enter a recipe name (e.g., `Chocolate Lava Cake`) to find culinary cousins.

### 3. Strict Match Mode
- **Exact Ingredient Count**: A new toggle that filters results to only include recipes with the **exact same number of ingredients** as your input. Perfect for when you want to use *only* what you listed.

### 4. Premium UI
- **Glassmorphism Design**: Custom CSS styling for a modern, clean look.
- **Interactive Cards**: Recipe results are displayed as beautiful cards with hover effects, "Match %" badges, and expandable details.
- **Responsive Grid**: Layout adapts to your screen size.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**: Frontend UI.
- **Scikit-Learn**: TF-IDF Vectorization and Nearest Neighbors model.
- **Pandas/Numpy**: Data manipulation.
- **Joblib**: Model persistence.

---

## 📂 Project Structure

```
recipe_recommender/
├── app/
│   ├── assets/          # Images and CSS
│   │   └── style.css    # Premium styling
│   └── streamlit_app.py # Main application entry point
├── data/
│   └── 1_Recipe_csv.csv # Dataset
├── models/              # Generated artifacts (TF-IDF matrix, vectorizer, etc.)
├── scripts/
│   ├── build_artifacts.py # Script to train/build models
│   └── analyze_data.py    # (Optional) Data analysis tool
├── src/
│   ├── recommender.py   # RecipeRecommender class (Hybrid Logic)
│   └── utils.py         # Text parsing and helper functions
├── README.md
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Create virtual env
python -m venv venv
# Activate (Windows)
.\venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Models
Before running the app for the first time, you must generate the TF-IDF matrices and metadata.
```bash
python scripts/build_artifacts.py
```
*This will create files in the `models/` directory.*

### 4. Run the App
```bash
streamlit run app/streamlit_app.py
```
*Note: The first run might take a few seconds to cache the ingredient parser.*

---

## 🔍 How to Use

1.  **Launch the App**: Open the URL provided by Streamlit (usually `http://localhost:8501`).
2.  **Choose a Mode**:
    *   **By Ingredients**: Type ingredients separated by commas.
    *   **Similar Recipe**: Type the name of a recipe you like.
3.  **Refine**:
    *   Use the **"Strict Ingredient Count"** checkbox for exact matches.
    *   Use filters for Category or Subcategory.
4.  **Explore**: Click on a recipe card to expand it and view the full instructions.

---

## 🧠 Model Details

The recommender uses a **Hybrid Scoring Formula**:
```python
Final Score = (0.4 * TF-IDF_Similarity) + (0.6 * Ingredient_Overlap)
```
- **TF-IDF**: Captures semantic similarity in text.
- **Overlap**: `intersection(user_ingredients, recipe_ingredients) / len(user_ingredients)`
- This ensures that while we find similar recipes, we heavily bias towards those that actually utilize your provided ingredients.

---

## 📝 License
This project is for educational purposes.
