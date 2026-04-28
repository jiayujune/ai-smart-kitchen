import pandas as pd
import ast
from models.receipe import Recipe

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []  # If parsing fails, return an empty list

def load_data(input_path):
    df = pd.read_csv(input_path)

    df["ingredients"] = df["ingredients"].apply(
        lambda x: safe_eval(x) if isinstance(x, str) else x
    )

    recipes = []

    for row in df.to_dict("records"):
        recipes.append(
            Recipe(
                name=row["name"],
                calories=row["calories"],
                protein=row.get("protein", 0),
                ingredients=row["ingredients"],
                diet=row.get("diet", "any"),
                meal_type=row.get("meal_type", "dinner")
            )
        )

    return recipes