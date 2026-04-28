import os
import pandas as pd
import ast

from preprocessing.data_cleaning import clean_data
from kitchen_csp.csp_solver import CSPSolver
from features.nutrition import analyze
from features.grocery import generate


def load_data(path):
    df = pd.read_csv(path)

    df["ingredients"] = df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    return df.to_dict("records")


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_path = os.path.join(BASE_DIR, "data/raw/RAW_recipes.csv")
    processed_path = os.path.join(BASE_DIR, "data/processed/cleaned_recipes.csv")

    # Step 1: Clean data
    clean_data(raw_path, processed_path)

    # Step 2: Load cleaned data
    recipes = load_data(processed_path)

    # Step 3: Define constraints
    constraints = {
        "max_calories": 2000,
        "diet": "vegetarian"
    }

    # Step 4: Solve CSP
    solver = CSPSolver(recipes, constraints)
    solution = solver.backtrack()

    # Step 5: Output
    if solution:
        print("\n🍽 Meal Plan:")
        for meal, recipe in solution.items():
            print(f"{meal}: {recipe['name']}")

        print("\n🥗 Nutrition:")
        print(analyze(solution))

        print("\n🛒 Grocery List:")
        print(generate(solution))
    else:
        print("❌ No solution found")


if __name__ == "__main__":
    main()