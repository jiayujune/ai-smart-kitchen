import json
import re
from typing import Dict, List, Optional, Set, Tuple


class SmartKitchenRecommender:
    def __init__(self, recipe_file: str):
        self.recipes = self.load_recipes(recipe_file)
        self.recipe_lookup = {recipe.get("id"): recipe for recipe in self.recipes if recipe.get("id") is not None}
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
            "spring onion": "onion",
            "spring onions": "onion",
            "scallion": "onion",
            "scallions": "onion",
            "white rice": "rice",
            "brown rice": "rice",
            "arborio rice": "rice",
            "jasmine rice": "rice",
            "basmati rice": "rice",
            "eggs": "egg",
            "garlic cloves": "garlic",
            "garlic clove": "garlic",
            "bell peppers": "bell pepper",
            "red bell pepper": "bell pepper",
            "green bell pepper": "bell pepper",
            "cheddar cheese": "cheese",
            "mozzarella cheese": "cheese",
            "parmesan cheese": "cheese",
            "olive oil": "oil",
            "vegetable oil": "oil",
            "canola oil": "oil",
            "chicken breasts": "chicken breast",
            "boneless skinless chicken breast": "chicken breast",
            "boneless skinless chicken breasts": "chicken breast",
            "ground beef": "beef",
            "lean ground beef": "beef",
            "all purpose flour": "flour",
            "plain flour": "flour",
        }
        self.pantry_staples = {
            "salt",
            "pepper",
            "black pepper",
            "oil",
            "water",
            "flour",
            "sugar",
            "butter",
        }

    def resolve_pantry_staples(self, user_profile: Optional[Dict]) -> Set[str]:
        if user_profile and user_profile.get("pantry_staples"):
            return set(user_profile["pantry_staples"])
        return set(self.pantry_staples)

    def load_recipes(self, recipe_file: str) -> List[Dict]:
        with open(recipe_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_recipe_by_id(self, recipe_id: int) -> Optional[Dict]:
        return self.recipe_lookup.get(recipe_id)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        if text in self.ingredient_map:
            return self.ingredient_map[text]

        singularized = self.singularize_token(text)
        return self.ingredient_map.get(singularized, singularized)

    def singularize_token(self, text: str) -> str:
        if len(text) <= 3:
            return text

        if text.endswith("ies") and len(text) > 4:
            return f"{text[:-3]}y"

        if text.endswith("es") and not text.endswith(("ses", "xes", "zes")):
            return text[:-2]

        if text.endswith("s") and not text.endswith("ss"):
            return text[:-1]

        return text

    def normalize_ingredients(self, ingredients: List[str]) -> Set[str]:
        normalized = set()
        for item in ingredients:
            if isinstance(item, str) and item.strip():
                normalized_item = self.normalize_text(item)
                if normalized_item:
                    normalized.add(normalized_item)
        return normalized

    def compute_match_details(
        self,
        user_ingredients: Set[str],
        recipe_ingredients: Set[str],
        pantry_staples: Set[str],
    ) -> Tuple[float, float, List[str], List[str], List[str]]:
        matched = sorted(user_ingredients.intersection(recipe_ingredients))
        staple_matches = sorted(item for item in matched if item in pantry_staples)
        missing = sorted(
            item
            for item in recipe_ingredients.difference(user_ingredients)
            if item not in pantry_staples
        )

        if not recipe_ingredients:
            return 0.0, 0.0, matched, missing, staple_matches

        essential_ingredients = recipe_ingredients.difference(pantry_staples)
        if not essential_ingredients:
            essential_ingredients = recipe_ingredients

        coverage = len(user_ingredients.intersection(essential_ingredients)) / len(essential_ingredients)
        overlap = len(matched) / len(user_ingredients) if user_ingredients else 0.0
        return coverage, overlap, matched, missing, staple_matches

    def is_reasonable_recipe_name(self, name: str) -> bool:
        if not name or len(name.strip()) < 3:
            return False

        bad_phrases = ["no name", "recipe", "1 1 1", "2 4 6 8"]
        lower_name = name.lower()
        return not any(phrase in lower_name for phrase in bad_phrases)

    def extract_calories(self, nutrition: List) -> Optional[float]:
        if isinstance(nutrition, list) and nutrition:
            calories = nutrition[0]
            if isinstance(calories, (int, float)):
                return round(float(calories), 1)
        return None

    def build_health_label(self, calories: Optional[float]) -> str:
        if calories is None:
            return "Unknown"
        if calories <= 300:
            return "Light"
        if calories <= 600:
            return "Balanced"
        return "Hearty"

    def build_explanation(
        self,
        matched_count: int,
        missing_count: int,
        health_label: str,
        matched: List[str],
        missing: List[str],
        staple_matches: List[str],
        personalization_reasons: List[str],
    ) -> str:
        if matched_count >= 4 and missing_count <= 2:
            fit_message = "Strong pantry fit with only a few extra items needed."
        elif matched_count >= 3:
            fit_message = "Good ingredient overlap for a practical weeknight option."
        else:
            fit_message = "Partial ingredient fit if you are open to adding a few items."

        reason_parts = []
        if matched:
            top_matches = ", ".join(matched[:3])
            reason_parts.append(f"Works well with {top_matches}")
        if missing:
            top_missing = ", ".join(missing[:2])
            reason_parts.append(f"you only need {top_missing}")
        if staple_matches:
            top_staples = ", ".join(staple_matches[:2])
            reason_parts.append(f"and uses your pantry staples like {top_staples}")
        if personalization_reasons:
            reason_parts.append(f"Personalized because it was {', '.join(personalization_reasons)}")

        reason_text = ". ".join(reason_parts)
        if reason_text:
            return f"{fit_message} {reason_text}. {health_label} calorie level."
        return f"{fit_message} {health_label} calorie level."

    def compute_base_score(
        self,
        coverage: float,
        overlap: float,
        matched_count: int,
        missing_count: int,
        calories: Optional[float],
    ) -> float:
        calorie_bonus = 0.0
        if calories is not None:
            if calories <= 350:
                calorie_bonus = 0.12
            elif calories <= 600:
                calorie_bonus = 0.06
            elif calories > 800:
                calorie_bonus = -0.18

        return round(
            coverage * 0.55
            + overlap * 0.20
            + matched_count * 0.12
            - missing_count * 0.08
            + calorie_bonus,
            4,
        )

    def compute_personalization_bonus(
        self,
        recipe: Dict,
        recipe_ingredients: Set[str],
        user_profile: Optional[Dict],
    ) -> Tuple[float, List[str]]:
        if not user_profile:
            return 0.0, []

        recipe_id = recipe.get("id")
        liked = set(user_profile.get("liked_recipes", []))
        favorites = set(user_profile.get("favorites", []))
        feedback = user_profile.get("recipe_feedback", {}).get(str(recipe_id), {})
        ingredient_counts = user_profile.get("ingredient_counts", {})
        total_pref = sum(ingredient_counts.values()) or 1

        bonus = 0.0
        reasons = []

        if recipe_id in favorites:
            bonus += 0.35
            reasons.append("favorited before")

        if recipe_id in liked:
            bonus += 0.22
            reasons.append("liked before")

        view_bonus = min(feedback.get("views", 0), 5) * 0.02
        bonus += view_bonus
        if feedback.get("views", 0) >= 2:
            reasons.append("seen often")

        preference_strength = sum(ingredient_counts.get(item, 0) for item in recipe_ingredients) / total_pref
        if preference_strength > 0:
            ingredient_bonus = min(preference_strength * 0.65, 0.28)
            bonus += ingredient_bonus
            if ingredient_bonus >= 0.1:
                reasons.append("matches your frequent ingredients")

        return round(bonus, 4), reasons[:2]

    def recommend_recipes(
        self,
        user_ingredients: List[str],
        top_k: int = 5,
        min_score: float = 0.3,
        min_matched_count: int = 2,
        max_missing_count: int = 5,
        user_profile: Optional[Dict] = None,
        sort_mode: str = "best_match",
    ) -> List[Dict]:
        normalized_user_ingredients = self.normalize_ingredients(user_ingredients)
        if not normalized_user_ingredients:
            return []

        recommendations = []
        pantry_staples = self.resolve_pantry_staples(user_profile)
        liked = set(user_profile.get("liked_recipes", [])) if user_profile else set()
        favorites = set(user_profile.get("favorites", [])) if user_profile else set()

        for recipe in self.recipes:
            recipe_name = recipe.get("name", "unknown recipe")
            recipe_id = recipe.get("id")

            if not self.is_reasonable_recipe_name(recipe_name):
                continue

            recipe_ingredients = self.normalize_ingredients(recipe.get("ingredients", []))
            coverage, overlap, matched, missing, staple_matches = self.compute_match_details(
                normalized_user_ingredients,
                recipe_ingredients,
                pantry_staples,
            )

            matched_count = len(matched)
            missing_count = len(missing)
            calories = self.extract_calories(recipe.get("nutrition", []))

            if calories is not None and calories > 900:
                continue

            if coverage < min_score or matched_count < min_matched_count or missing_count > max_missing_count:
                continue

            base_score = self.compute_base_score(
                coverage=coverage,
                overlap=overlap,
                matched_count=matched_count,
                missing_count=missing_count,
                calories=calories,
            )
            personalization_bonus, personalization_reasons = self.compute_personalization_bonus(
                recipe=recipe,
                recipe_ingredients=recipe_ingredients,
                user_profile=user_profile,
            )
            final_score = round(base_score + personalization_bonus, 4)
            health_label = self.build_health_label(calories)

            recommendations.append(
                {
                    "id": recipe_id,
                    "name": recipe_name,
                    "score": round(coverage, 3),
                    "overlap_score": round(overlap, 3),
                    "matched_ingredients": matched,
                    "missing_ingredients": missing,
                    "total_ingredients": len(recipe_ingredients),
                    "matched_count": matched_count,
                    "missing_count": missing_count,
                    "calories": calories,
                    "health_label": health_label,
                    "base_score": base_score,
                    "personalization_bonus": personalization_bonus,
                    "final_score": final_score,
                    "match_percent": round(coverage * 100),
                    "is_liked": recipe_id in liked,
                    "is_favorite": recipe_id in favorites,
                    "explanation": self.build_explanation(
                        matched_count=matched_count,
                        missing_count=missing_count,
                        health_label=health_label,
                        matched=matched,
                        missing=missing,
                        staple_matches=staple_matches,
                        personalization_reasons=personalization_reasons,
                    ),
                }
            )

        recommendations = self.sort_recommendations(recommendations, sort_mode)

        return recommendations[:top_k]

    def sort_recommendations(self, recommendations: List[Dict], sort_mode: str) -> List[Dict]:
        if sort_mode == "fewest_missing":
            recommendations.sort(
                key=lambda item: (
                    item["missing_count"],
                    -item["matched_count"],
                    -item["final_score"],
                    -item["calories"] if item["calories"] is not None else 0,
                )
            )
            return recommendations

        if sort_mode == "lowest_calories":
            recommendations.sort(
                key=lambda item: (
                    item["calories"] if item["calories"] is not None else float("inf"),
                    -item["matched_count"],
                    -item["final_score"],
                )
            )
            return recommendations

        if sort_mode == "most_personalized":
            recommendations.sort(
                key=lambda item: (
                    item["personalization_bonus"],
                    item["final_score"],
                    item["matched_count"],
                    -item["missing_count"],
                ),
                reverse=True,
            )
            return recommendations

        recommendations.sort(
            key=lambda item: (
                item["final_score"],
                item["matched_count"],
                -item["missing_count"],
                item["overlap_score"],
            ),
            reverse=True,
        )
        return recommendations


if __name__ == "__main__":
    recommender = SmartKitchenRecommender("recipes_clean.json")
    user_items = ["egg", "tomato", "rice", "onion"]
    results = recommender.recommend_recipes(
        user_items,
        top_k=10,
        min_score=0.15,
        min_matched_count=2,
        max_missing_count=6,
    )

    print("User ingredients:", user_items)
    print("\nTop recommendations:\n")

    if not results:
        print("No strong recipe matches found.")
    else:
        for index, recipe in enumerate(results, start=1):
            print(f"{index}. {recipe['name']}")
            print(f"   Match percent: {recipe['match_percent']}%")
            print(f"   Coverage score: {recipe['score']}")
            print(f"   Pantry overlap: {recipe['overlap_score']}")
            print(f"   Personalization bonus: {recipe['personalization_bonus']}")
            print(f"   Matched count: {recipe['matched_count']}")
            print(f"   Missing count: {recipe['missing_count']}")
            print(f"   Matched: {recipe['matched_ingredients']}")
            print(f"   Missing: {recipe['missing_ingredients']}")
            print(f"   Calories: {recipe['calories']}")
            print(f"   Why this recipe: {recipe['explanation']}")
            print()
