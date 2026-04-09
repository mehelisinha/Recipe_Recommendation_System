import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.recommender import RecipeRecommender
    print("✅ Successfully imported RecipeRecommender")
except ImportError as e:
    print(f"❌ Failed to import RecipeRecommender: {e}")
    sys.exit(1)

def verify():
    print("Initializing Recommender...")
    try:
        # Simulate how streamlit_app loads it: model_dir relative to __file__ -> .. -> models
        # But here we are in scripts/, so .. -> models is correct too.
        # Streamlit app is in app/, so .. -> models is also correct.
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        rec = RecipeRecommender(model_dir)
        print("✅ Models loaded successfully")
        
        # Test Search
        results = rec.search("tomato", top_k=5)
        if not results.empty:
            print(f"✅ Search returned {len(results)} results for 'tomato'")
        else:
            print("⚠️ Search returned empty results (might be valid if no tomato recipes, but unusual)")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
