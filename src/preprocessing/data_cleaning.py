import pandas as pd
import ast


def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Original columns:", df.columns)

    df = df[["name", "ingredients", "nutrition"]]
    df = df.dropna()

    df["ingredients"] = df["ingredients"].apply(safe_eval)
    df["nutrition"] = df["nutrition"].apply(safe_eval)

    df["calories"] = df["nutrition"].apply(lambda x: x[0] if len(x) > 0 else 0)
    df["protein"] = df["nutrition"].apply(lambda x: x[4] if len(x) > 4 else 0)

    #Save cleaned data
    df.to_csv(output_path, index=False)

    print(f" Cleaned data saved to {output_path}")

def load_data(path):
    df = pd.read_csv(path)

    df["ingredients"] = df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    return df.to_dict("records")

def detect_diet(ingredients):
    non_veg_keywords = [
        "chicken", "beef", "pork", "fish",
        "meat", "crab", "turkey", "lamb"
    ]

    for ingredient in ingredients:
        if any(word in ingredient.lower() for word in non_veg_keywords):
            return "non-veg"

    return "vegetarian"
