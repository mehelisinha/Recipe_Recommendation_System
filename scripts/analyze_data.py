import pandas as pd
import ast
import os

def load_data():
    # Try loading from data directory
    csv_path = os.path.join("data", "1_Recipe_csv.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def analyze():
    df = load_data()
    if df is None:
        return

    print(f"Total recipes: {len(df)}")
    
    # Check if 'ingredients' column exists
    if 'ingredients' not in df.columns:
        print("No 'ingredients' column found.")
        return

    # Parse ingredients
    # They seem to be stringified lists "['a', 'b']"
    def count_ingredients(ing_str):
        try:
            # simple parse
            if isinstance(ing_str, str) and ing_str.strip().startswith("["):
                l = ast.literal_eval(ing_str)
                return len(l)
            return 0
        except:
            return 0

    df['ing_count'] = df['ingredients'].apply(count_ingredients)
    
    print("\n--- Ingredient Count Statistics ---")
    print(df['ing_count'].describe())
    
    print("\n--- Low Ingredient Counts ---")
    for i in range(1, 6):
        count = len(df[df['ing_count'] == i])
        print(f"Recipes with exactly {i} ingredients: {count}")

    print("\n--- Example 2-ingredient recipes ---")
    two_ing = df[df['ing_count'] == 2].head(5)
    for idx, row in two_ing.iterrows():
        print(f"- {row['recipe_title']}: {row['ingredients']}")

if __name__ == "__main__":
    analyze()
