import re
import ast

# ---------------------------------------------------------------------
# Clean user-entered ingredient text
# ---------------------------------------------------------------------
def clean_user_ingredients(text: str):
    """
    Clean user-entered ingredient text: remove extra whitespace,
    normalize commas and line breaks.
    """
    text = re.sub(r"[\n\r]+", " ", str(text).strip())
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


# ---------------------------------------------------------------------
# Helpers for pretty-printing list-like fields
# ---------------------------------------------------------------------
def _to_list(value):
    """
    Safely convert strings like "['item1', 'item2']" into real Python lists.
    Returns [] if conversion fails.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Looks like a Python list? Try to parse.
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
        return [s] if s else []
    return []


def _fix_mojibake(s: str) -> str:
    """
    Automatically repair mojibake produced when UTF-8 text is wrongly decoded as latin-1.

    Example:
      'Â¼' -> '¼'
      'â teaspoon salt' -> '⅛ teaspoon salt'

    If the string is already fine, it is returned unchanged.
    """
    try:
        # Interpret the existing text as latin1 bytes, then decode as utf-8.
        # This reverses common mojibake without hardcoding replacements.
        return s.encode("latin1").decode("utf-8")
    except Exception:
        return s


def pretty_list(value, bullet: bool = True) -> str:
    """
    Convert list-like '["¼ teaspoon salt", ...]' into a readable string.

    bullet=True  -> one item per line with '• ' bullets
    bullet=False -> comma-separated line
    """
    raw_items = _to_list(value)
    items = [_fix_mojibake(str(x)).strip() for x in raw_items]

    if not items:
        return ""

    if bullet:
        return "• " + "\n• ".join(items)
    return ", ".join(items)
