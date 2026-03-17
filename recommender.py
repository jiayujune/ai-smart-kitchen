import json
import re
from typing import List, Dict, Tuple


class SmartKitchenRecommender:
    def __init__(self, recipe_file: str):
        self.recipes = self.load_recipes(recipe_file)
        self.ingredient_map = {
            "roma tomato": "tomato",
            "roma tomatoes": "tomato",
            "cherry tomato": "tomato",
            "cherry tomatoes": "tomato",
            "tomatoes": "tomato",
            "yellow onion": "onion",
            "red onion": "onion",
            "green onion": "onion",
            "green onions": "onion",
            "scallion": "onion",
            "scallions": "onion",
            "white rice": "rice",
            "brown rice": "rice",
            "arborio rice": "rice",
            "jasmine rice": "rice",
            "basmati rice": "rice",
            "eggs": "egg"
        }

    def load_recipes(self, recipe_file: str) -> List[Dict]:
        with open(recipe_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)

        if text in self.ingredient_map:
            text = self.ingredient_map[text]

        return text

    def normalize_ingredients(self, ingredients: List[str]) -> set:
        return {self.normalize_text(item) for item in ingredients if isinstance(item, str) and item.strip()}

    def compute_match_score(
        self,
        user_ingredients: set,
        recipe_ingredients: set
    ) -> Tuple[float, List[str], List[str]]:
        matched = list(user_ingredients.intersection(recipe_ingredients))
        missing = list(recipe_ingredients.difference(user_ingredients))

        if len(recipe_ingredients) == 0:
            return 0.0, matched, missing

        score = len(matched) / len(recipe_ingredients)
        return score, matched, missing

    def is_reasonable_recipe_name(self, name: str) -> bool:
        if not name or len(name.strip()) < 3:
            return False

        bad_phrases = [
            "no name",
            "recipe",
            "1 1 1",
            "2 4 6 8"
        ]

        lower_name = name.lower()
        for phrase in bad_phrases:
            if phrase in lower_name:
                return False

        return True

    def recommend_recipes(
        self,
        user_ingredients: List[str],
        top_k: int = 5,
        min_score: float = 0.3,
        min_matched_count: int = 2,
        max_missing_count: int = 5
    ) -> List[Dict]:
        normalized_user_ingredients = self.normalize_ingredients(user_ingredients)
        recommendations = []

        for recipe in self.recipes:
            recipe_name = recipe.get("name", "unknown recipe")
            recipe_id = recipe.get("id", None)

            if not self.is_reasonable_recipe_name(recipe_name):
                continue

            recipe_ingredients = self.normalize_ingredients(recipe.get("ingredients", []))
            recipe_nutrition = recipe.get("nutrition", [])

            score, matched, missing = self.compute_match_score(
                normalized_user_ingredients,
                recipe_ingredients
            )

            matched_count = len(matched)
            missing_count = len(missing)

            calories = None
            if isinstance(recipe_nutrition, list) and len(recipe_nutrition) > 0:
                calories = recipe_nutrition[0]

            if calories is not None and calories > 800:
                continue

            if (
                score >= min_score
                and matched_count >= min_matched_count
                and missing_count <= max_missing_count
            ):
                final_score = matched_count * 2 - missing_count

                recommendations.append({
                    "id": recipe_id,
                    "name": recipe_name,
                    "score": round(score, 3),
                    "matched_ingredients": sorted(matched),
                    "missing_ingredients": sorted(missing),
                    "total_ingredients": len(recipe_ingredients),
                    "matched_count": matched_count,
                    "missing_count": missing_count,
                    "calories": calories,
                    "final_score": final_score
                })

        # better ranking:
        # 1. higher matched_count
        # 2. higher score
        # 3. fewer missing ingredients
        recommendations.sort(
            key=lambda x: (
                x["matched_count"],
                x["final_score"],
                x["score"]
            ),
            reverse=True
        )

        return recommendations[:top_k]


if __name__ == "__main__":
    recommender = SmartKitchenRecommender("recipes_clean.json")

    user_items = ["egg", "tomato", "rice", "onion"]
    results = recommender.recommend_recipes(
        user_items,
        top_k=10,
        min_score=0.15,
        min_matched_count=2,
        max_missing_count=6
    )

    print("User ingredients:", user_items)
    print("\nTop recommendations:\n")

    if not results:
        print("No strong recipe matches found.")
    else:
        for i, recipe in enumerate(results, start=1):
            print(f"{i}. {recipe['name']}")
            print(f"   Score: {recipe['score']}")
            print(f"   Matched count: {recipe['matched_count']}")
            print(f"   Missing count: {recipe['missing_count']}")
            print(f"   Matched: {recipe['matched_ingredients']}")
            print(f"   Missing: {recipe['missing_ingredients']}")
            print(f"   Calories: {recipe['calories']}")
            print()