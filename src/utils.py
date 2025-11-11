import re

def clean_user_ingredients(text: str):
    text = re.sub(r"[\\n\\r]+", " ", str(text).strip())
    text = re.sub(r"\\s*,\\s*", ", ", text)
    return text