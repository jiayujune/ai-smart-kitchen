import pandas as pd
import ast
import json
import re

print("script started")


def safe_literal_eval(value):
    """
    Safely convert string representation of list into Python list.
    Example:
    "['egg', 'tomato']" -> ['egg', 'tomato']
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []


def normalize_ingredient(text):
    """
    Normalize ingredient text:
    - lowercase
    - remove extra spaces
    - remove special characters except spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_ingredient_list(ingredient_list):
    """
    Clean ingredient list and remove empty values.
    """
    cleaned = []
    for item in ingredient_list:
        normalized = normalize_ingredient(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def main():
    input_file = "RAW_recipes.csv"
    output_file = "recipes_clean.json"

    # 1. Load raw data
    df = pd.read_csv(input_file)

    # 2. Keep only useful columns
    needed_cols = ["id", "name", "ingredients", "nutrition", "steps", "n_ingredients"]
    df = df[needed_cols]

    # 3. Drop rows with missing key fields
    df = df.dropna(subset=["id", "name", "ingredients"])

    # 4. Parse ingredients / nutrition / steps
    df["ingredients"] = df["ingredients"].apply(safe_literal_eval)
    df["nutrition"] = df["nutrition"].apply(safe_literal_eval)
    df["steps"] = df["steps"].apply(safe_literal_eval)

    # 5. Clean ingredient text
    df["ingredients"] = df["ingredients"].apply(clean_ingredient_list)

    # 6. Remove rows with empty ingredient lists
    df = df[df["ingredients"].map(len) > 0]

    # 7. Optional: filter out recipes with too few or too many ingredients
    df = df[(df["n_ingredients"] >= 2) & (df["n_ingredients"] <= 20)]

    # 8. Optional: only keep a subset for first prototype
    df = df.head(5000)

    # 9. Convert to list of dictionaries
    cleaned_recipes = []
    for _, row in df.iterrows():
        cleaned_recipes.append({
            "id": int(row["id"]),
            "name": str(row["name"]).strip().lower(),
            "ingredients": row["ingredients"],
            "nutrition": row["nutrition"],
            "steps": row["steps"],
            "n_ingredients": int(row["n_ingredients"])
        })

    # 10. Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_recipes, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved {len(cleaned_recipes)} recipes to {output_file}")


if __name__ == "__main__":
    main()