import re
import ast
from typing import List, Set

# ---------------------------------------------------------------------
# Text Cleaning & Parsing
# ---------------------------------------------------------------------

def clean_text_list(text_or_list) -> List[str]:
    """
    Parses a string representation of a list (e.g. "['a', 'b']") into a real python list.
    Safely handles NA or malformed strings.
    """
    if not text_or_list:
        return []
    
    if isinstance(text_or_list, list):
        return [str(x) for x in text_or_list]
    
    if isinstance(text_or_list, str):
        s = text_or_list.strip()
        # Basic check for list-like string
        if s.startswith("[") and s.endswith("]"):
            try:
                # ast.literal_eval is safer than eval
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                pass
        # Fallback: comma separated?
        return [x.strip() for x in s.split(",") if x.strip()]
        
    return []

def normalize_ingredient(ing: str) -> str:
    """
    Normalizes a single ingredient string for matching.
    Input: "1/2 cup chopped fresh parsley"
    Output: "chopped fresh parsley"  (Simplified for now, just lowercasing + strip)
    """
    # A true parser is complex, for now we just lowercase and remove simple punctuation
    return ing.lower().strip().strip(".,")

def clean_user_query_ingredients(query: str) -> Set[str]:
    """
    Split user query by commas and normalize to a set of keywords.
    """
    raw_tokens = [x.strip() for x in query.split(",")]
    return {normalize_ingredient(t) for t in raw_tokens if t.strip()}

def clean_user_ingredients(text: str):
    """
    Legacy wrapper for compatibility if needed.
    """
    return text

# ---------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------

def _fix_mojibake(s: str) -> str:
    """
    Automatically repair mojibake produced when UTF-8 text is wrongly decoded as latin-1.
    """
    try:
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s

def pretty_list(value, bullet: bool = True) -> str:
    """
    Variable input -> bulleted string
    """
    items = clean_text_list(value)
    items = [_fix_mojibake(x).strip() for x in items]
    
    if not items:
        return ""
    
    if bullet:
        return "<br>".join([f"• {x}" for x in items])
    return ", ".join(items)
