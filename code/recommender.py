import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ml_model import attach_ml_prediction, load_ml_ranker


class SmartKitchenRecommender:
    def __init__(self, recipe_file: str):
        self.dietary_rules = {
            "vegetarian": {
                "label": "Vegetarian",
                "blocked_terms": {
                    "anchovy",
                    "bacon",
                    "beef",
                    "beef broth",
                    "beef stock",
                    "chicken",
                    "chicken broth",
                    "chicken stock",
                    "clam",
                    "crab",
                    "duck",
                    "fish",
                    "fish stock",
                    "gelatin",
                    "ham",
                    "lamb",
                    "lobster",
                    "meat",
                    "oyster",
                    "pepperoni",
                    "pork",
                    "prosciutto",
                    "salami",
                    "salmon",
                    "sausage",
                    "shrimp",
                    "tuna",
                    "turkey",
                    "veal",
                },
            },
            "gluten_free": {
                "label": "Gluten Free",
                "blocked_terms": {
                    "barley",
                    "beer",
                    "bread",
                    "breadcrumb",
                    "bulgur",
                    "couscous",
                    "cracker",
                    "flour",
                    "lasagna",
                    "malt",
                    "noodle",
                    "panko",
                    "pasta",
                    "rye",
                    "seitan",
                    "soy sauce",
                    "spaghetti",
                    "tortilla",
                    "wheat",
                },
            },
            "vegan": {
                "label": "Vegan",
                "blocked_terms": {
                    "anchovy",
                    "bacon",
                    "beef",
                    "beef broth",
                    "beef stock",
                    "butter",
                    "buttermilk",
                    "casein",
                    "cheese",
                    "chicken",
                    "chicken broth",
                    "chicken stock",
                    "clam",
                    "cream",
                    "crab",
                    "duck",
                    "egg",
                    "fish",
                    "fish stock",
                    "gelatin",
                    "ghee",
                    "ham",
                    "honey",
                    "lamb",
                    "lobster",
                    "mayonnaise",
                    "meat",
                    "milk",
                    "oyster",
                    "pepperoni",
                    "pork",
                    "prosciutto",
                    "salami",
                    "salmon",
                    "sausage",
                    "shrimp",
                    "tuna",
                    "turkey",
                    "veal",
                    "whey",
                    "yogurt",
                },
            },
            "dairy_free": {
                "label": "Dairy Free",
                "blocked_terms": {
                    "butter",
                    "buttermilk",
                    "casein",
                    "cheese",
                    "cream",
                    "creme fraiche",
                    "ghee",
                    "half and half",
                    "ice cream",
                    "milk",
                    "parmesan",
                    "sour cream",
                    "whey",
                    "yogurt",
                },
            },
            "nut_free": {
                "label": "Nut Free",
                "blocked_terms": {
                    "almond",
                    "brazil nut",
                    "cashew",
                    "chestnut",
                    "hazelnut",
                    "macadamia",
                    "nut",
                    "peanut",
                    "pecan",
                    "pine nut",
                    "pistachio",
                    "walnut",
                },
            },
        }
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
            "heavy cream": "cream",
            "sour cream": "sour cream",
            "greek yogurt": "yogurt",
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
            "wheat flour": "flour",
            "whole wheat flour": "flour",
            "peanuts": "peanut",
            "almonds": "almond",
            "walnuts": "walnut",
            "pecans": "pecan",
            "cashews": "cashew",
            "pistachios": "pistachio",
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
        self.substitution_map = {
            "butter": ["oil", "olive oil", "margarine", "ghee", "applesauce"],
            "oil": ["olive oil", "vegetable oil", "canola oil", "butter"],
            "olive oil": ["oil", "vegetable oil", "canola oil", "butter"],
            "vegetable oil": ["oil", "olive oil", "canola oil", "butter"],
            "egg": ["ground flaxseed", "chia seed gel", "applesauce", "banana", "yogurt"],
            "milk": ["almond milk", "soy milk", "oat milk", "water", "cream"],
            "cream": ["milk", "half and half", "evaporated milk", "coconut milk", "greek yogurt"],
            "sour cream": ["greek yogurt", "plain yogurt", "cream cheese", "coconut yogurt"],
            "yogurt": ["sour cream", "greek yogurt", "buttermilk", "coconut yogurt"],
            "cheese": ["mozzarella cheese", "cheddar cheese", "parmesan cheese", "nutritional yeast"],
            "parmesan cheese": ["cheese", "pecorino", "nutritional yeast"],
            "flour": ["whole wheat flour", "all purpose flour", "cornstarch", "oat flour"],
            "bread": ["tortilla", "pita", "cracker", "rice"],
            "pasta": ["rice", "noodle", "zucchini noodle", "spaghetti squash"],
            "rice": ["quinoa", "couscous", "pasta", "cauliflower rice"],
            "tomato": ["tomato sauce", "tomato paste", "canned tomato", "red bell pepper"],
            "tomato sauce": ["tomato paste", "canned tomato", "tomato"],
            "onion": ["shallot", "green onion", "leek", "onion powder"],
            "garlic": ["garlic powder", "shallot", "onion"],
            "bell pepper": ["red bell pepper", "green bell pepper", "poblano", "zucchini"],
            "carrot": ["sweet potato", "parsnip", "bell pepper"],
            "potato": ["sweet potato", "cauliflower", "turnip"],
            "chicken breast": ["chicken thigh", "turkey breast", "tofu", "chickpea"],
            "chicken": ["chicken breast", "chicken thigh", "turkey", "tofu"],
            "beef": ["ground beef", "turkey", "mushroom", "lentil"],
            "ground beef": ["beef", "ground turkey", "mushroom", "lentil"],
            "bacon": ["ham", "turkey bacon", "smoked paprika", "mushroom"],
            "shrimp": ["chicken", "tofu", "scallop", "fish"],
            "fish": ["salmon", "tuna", "shrimp", "tofu"],
            "sugar": ["honey", "maple syrup", "brown sugar"],
            "honey": ["maple syrup", "sugar", "agave"],
            "soy sauce": ["tamari", "coconut aminos", "salt"],
            "vinegar": ["lemon juice", "lime juice", "white wine vinegar"],
            "lemon juice": ["lime juice", "vinegar"],
            "peanut butter": ["almond butter", "sunflower seed butter", "tahini"],
            "peanut": ["almond", "cashew", "sunflower seed"],
            "almond": ["peanut", "cashew", "sunflower seed"],
        }
        self.substitution_filter_overrides = {
            "vegetarian": {
                "tofu",
                "chickpea",
                "mushroom",
                "lentil",
            },
            "vegan": {
                "almond milk",
                "soy milk",
                "oat milk",
                "coconut milk",
                "coconut yogurt",
                "ground flaxseed",
                "chia seed gel",
                "applesauce",
                "banana",
                "nutritional yeast",
                "tofu",
                "chickpea",
                "mushroom",
                "lentil",
                "sunflower seed butter",
                "tahini",
            },
            "dairy_free": {
                "almond milk",
                "soy milk",
                "oat milk",
                "coconut milk",
                "coconut yogurt",
                "nutritional yeast",
            },
        }
        self.recipes = self.load_recipes(recipe_file)
        self.recipe_lookup = {recipe.get("id"): recipe for recipe in self.recipes if recipe.get("id") is not None}
        self.recipe_ingredient_cache = self.build_recipe_ingredient_cache()
        self.ml_ranker = load_ml_ranker()

    def resolve_pantry_staples(self, user_profile: Optional[Dict]) -> Set[str]:
        if user_profile and user_profile.get("pantry_staples"):
            return set(user_profile["pantry_staples"])
        return set(self.pantry_staples)

    def load_recipes(self, recipe_file: str) -> List[Dict]:
        with open(recipe_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def build_recipe_ingredient_cache(self) -> Dict[int, Set[str]]:
        cache = {}
        for recipe in self.recipes:
            recipe_id = recipe.get("id")
            if recipe_id is None:
                continue
            cache[recipe_id] = self.normalize_ingredients(recipe.get("ingredients", []))
        return cache

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

    def normalize_dietary_filters(self, dietary_filters: Optional[List[str]]) -> Set[str]:
        if not dietary_filters:
            return set()
        return {item for item in dietary_filters if item in self.dietary_rules}

    def ingredient_contains_term(self, ingredient: str, term: str) -> bool:
        if ingredient == term:
            return True
        if " " in term:
            return term in ingredient
        return bool(re.search(rf"\b{re.escape(term)}\b", ingredient))

    def recipe_matches_dietary_filters(
        self,
        recipe_ingredients: Set[str],
        dietary_filters: Optional[List[str]],
    ) -> bool:
        active_filters = self.normalize_dietary_filters(dietary_filters)
        if not active_filters:
            return True

        for filter_key in active_filters:
            blocked_terms = self.dietary_rules[filter_key]["blocked_terms"]
            for ingredient in recipe_ingredients:
                if any(self.ingredient_contains_term(ingredient, term) for term in blocked_terms):
                    return False
        return True

    def dietary_labels_for_recipe(
        self,
        recipe_ingredients: Set[str],
        dietary_filters: Optional[List[str]],
    ) -> List[str]:
        return [
            self.dietary_rules[filter_key]["label"]
            for filter_key in self.normalize_dietary_filters(dietary_filters)
            if self.recipe_matches_dietary_filters(recipe_ingredients, [filter_key])
        ]

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

    def compute_knn_similarity(
        self,
        user_ingredients: Set[str],
        recipe_ingredients: Set[str],
        pantry_staples: Set[str],
    ) -> Tuple[float, float, float]:
        if not user_ingredients or not recipe_ingredients:
            return 0.0, 0.0, 0.0

        effective_recipe_ingredients = recipe_ingredients.difference(pantry_staples)
        if not effective_recipe_ingredients:
            effective_recipe_ingredients = recipe_ingredients

        matched_count = len(user_ingredients.intersection(effective_recipe_ingredients))
        if matched_count == 0:
            return 0.0, 0.0, 0.0

        cosine_similarity = matched_count / math.sqrt(
            len(user_ingredients) * len(effective_recipe_ingredients)
        )
        jaccard_similarity = matched_count / len(user_ingredients.union(effective_recipe_ingredients))
        knn_score = cosine_similarity * 0.8 + jaccard_similarity * 0.2
        return round(knn_score, 4), round(cosine_similarity, 4), round(jaccard_similarity, 4)

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

    def build_substitution_suggestions(
        self,
        missing: List[str],
        user_ingredients: Set[str],
        dietary_filters: Optional[List[str]] = None,
        limit: int = 4,
    ) -> List[Dict]:
        suggestions = []
        normalized_user = set(user_ingredients)

        for ingredient in missing:
            options = []
            seen = set()
            for raw_option in self.substitution_map.get(ingredient, []):
                option = self.normalize_text(raw_option)
                if not option or option == ingredient or option in seen:
                    continue
                if not self.substitution_matches_dietary_filters(option, dietary_filters):
                    continue

                seen.add(option)
                options.append(
                    {
                        "name": option,
                        "in_pantry": option in normalized_user,
                    }
                )

            if not options:
                continue

            options.sort(key=lambda item: (not item["in_pantry"], item["name"]))
            available = [item["name"] for item in options if item["in_pantry"]]
            suggestions.append(
                {
                    "ingredient": ingredient,
                    "options": options[:3],
                    "available_options": available[:3],
                    "has_pantry_match": bool(available),
                }
            )

            if len(suggestions) >= limit:
                break

        return suggestions

    def substitution_matches_dietary_filters(
        self,
        option: str,
        dietary_filters: Optional[List[str]],
    ) -> bool:
        for filter_key in self.normalize_dietary_filters(dietary_filters):
            if self.recipe_matches_dietary_filters({option}, [filter_key]):
                continue
            if option in self.substitution_filter_overrides.get(filter_key, set()):
                continue
            return False
        return True

    def compute_calorie_bonus(self, calories: Optional[float]) -> float:
        if calories is None:
            return 0.0
        if calories <= 350:
            return 0.12
        if calories <= 600:
            return 0.06
        if calories > 800:
            return -0.18
        return 0.0

    def compute_base_score(
        self,
        knn_score: float,
        coverage: float,
        overlap: float,
        matched_count: int,
        missing_count: int,
        calories: Optional[float],
    ) -> float:
        calorie_bonus = self.compute_calorie_bonus(calories)

        return round(
            knn_score * 0.45
            + coverage * 0.25
            + overlap * 0.10
            + matched_count * 0.08
            - missing_count * 0.05
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

    def build_score_breakdown(
        self,
        knn_score: float,
        coverage: float,
        overlap: float,
        matched_count: int,
        missing_count: int,
        calorie_bonus: float,
        personalization_bonus: float,
    ) -> List[Dict]:
        components = [
            {
                "label": "KNN similarity",
                "value": round(knn_score * 0.45, 4),
                "detail": f"{knn_score} x 0.45",
            },
            {
                "label": "Ingredient coverage",
                "value": round(coverage * 0.25, 4),
                "detail": f"{round(coverage, 3)} x 0.25",
            },
            {
                "label": "Pantry overlap",
                "value": round(overlap * 0.10, 4),
                "detail": f"{round(overlap, 3)} x 0.10",
            },
            {
                "label": "Matched items",
                "value": round(matched_count * 0.08, 4),
                "detail": f"{matched_count} x 0.08",
            },
            {
                "label": "Missing penalty",
                "value": round(-missing_count * 0.05, 4),
                "detail": f"-{missing_count} x 0.05",
            },
            {
                "label": "Calorie adjustment",
                "value": round(calorie_bonus, 4),
                "detail": "light/balanced bonus or high-calorie penalty",
            },
            {
                "label": "Personalization",
                "value": round(personalization_bonus, 4),
                "detail": "likes, saves, views, and frequent ingredients",
            },
        ]

        for component in components:
            value = component["value"]
            component["kind"] = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            component["bar_width"] = min(100, round(abs(value) / 0.45 * 100))
            component["signed_value"] = f"{value:+.4f}"

        return components

    def score_recipe(
        self,
        recipe: Dict,
        normalized_user_ingredients: Set[str],
        pantry_staples: Set[str],
        user_profile: Optional[Dict] = None,
        dietary_filters: Optional[List[str]] = None,
    ) -> Dict:
        recipe_id = recipe.get("id")
        recipe_ingredients = self.recipe_ingredient_cache.get(recipe_id) or self.normalize_ingredients(
            recipe.get("ingredients", [])
        )
        coverage, overlap, matched, missing, staple_matches = self.compute_match_details(
            normalized_user_ingredients,
            recipe_ingredients,
            pantry_staples,
        )
        knn_score, cosine_similarity, jaccard_similarity = self.compute_knn_similarity(
            normalized_user_ingredients,
            recipe_ingredients,
            pantry_staples,
        )

        matched_count = len(matched)
        missing_count = len(missing)
        calories = self.extract_calories(recipe.get("nutrition", []))
        calorie_bonus = self.compute_calorie_bonus(calories)
        base_score = self.compute_base_score(
            knn_score=knn_score,
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

        liked = set(user_profile.get("liked_recipes", [])) if user_profile else set()
        favorites = set(user_profile.get("favorites", [])) if user_profile else set()

        scored_recipe = {
            "id": recipe_id,
            "name": recipe.get("name", "unknown recipe"),
            "score": knn_score,
            "knn_score": knn_score,
            "cosine_similarity": cosine_similarity,
            "jaccard_similarity": jaccard_similarity,
            "coverage_score": round(coverage, 3),
            "overlap_score": round(overlap, 3),
            "matched_ingredients": matched,
            "missing_ingredients": missing,
            "substitution_suggestions": self.build_substitution_suggestions(
                missing=missing,
                user_ingredients=normalized_user_ingredients,
                dietary_filters=dietary_filters,
            ),
            "total_ingredients": len(recipe_ingredients),
            "matched_count": matched_count,
            "missing_count": missing_count,
            "calories": calories,
            "health_label": health_label,
            "calorie_bonus": calorie_bonus,
            "base_score": base_score,
            "personalization_bonus": personalization_bonus,
            "final_score": final_score,
            "match_percent": round(knn_score * 100),
            "is_liked": recipe_id in liked,
            "is_favorite": recipe_id in favorites,
            "dietary_labels": self.dietary_labels_for_recipe(recipe_ingredients, dietary_filters),
            "score_breakdown": self.build_score_breakdown(
                knn_score=knn_score,
                coverage=coverage,
                overlap=overlap,
                matched_count=matched_count,
                missing_count=missing_count,
                calorie_bonus=calorie_bonus,
                personalization_bonus=personalization_bonus,
            ),
            "score_formula": (
                "Final = 0.45*KNN + 0.25*coverage + 0.10*overlap + "
                "0.08*matched - 0.05*missing + calorie adjustment + personalization"
            ),
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
        return attach_ml_prediction(scored_recipe, self.ml_ranker)

    def recommend_recipes(
        self,
        user_ingredients: List[str],
        top_k: int = 5,
        min_score: float = 0.3,
        min_matched_count: int = 2,
        max_missing_count: int = 5,
        user_profile: Optional[Dict] = None,
        sort_mode: str = "best_match",
        dietary_filters: Optional[List[str]] = None,
    ) -> List[Dict]:
        normalized_user_ingredients = self.normalize_ingredients(user_ingredients)
        if not normalized_user_ingredients:
            return []

        recommendations = []
        pantry_staples = self.resolve_pantry_staples(user_profile)

        for recipe in self.recipes:
            recipe_name = recipe.get("name", "unknown recipe")

            if not self.is_reasonable_recipe_name(recipe_name):
                continue

            recipe_id = recipe.get("id")
            recipe_ingredients = self.recipe_ingredient_cache.get(recipe_id) or self.normalize_ingredients(
                recipe.get("ingredients", [])
            )
            if not self.recipe_matches_dietary_filters(recipe_ingredients, dietary_filters):
                continue

            scored_recipe = self.score_recipe(
                recipe=recipe,
                normalized_user_ingredients=normalized_user_ingredients,
                pantry_staples=pantry_staples,
                user_profile=user_profile,
                dietary_filters=dietary_filters,
            )

            if scored_recipe["calories"] is not None and scored_recipe["calories"] > 900:
                continue

            if (
                scored_recipe["knn_score"] < min_score
                or scored_recipe["matched_count"] < min_matched_count
                or scored_recipe["missing_count"] > max_missing_count
            ):
                continue

            recommendations.append(scored_recipe)

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

        if sort_mode == "ml_ranker":
            recommendations.sort(
                key=lambda item: (
                    item.get("ml_score", 0),
                    item["final_score"],
                    item["knn_score"],
                    -item["missing_count"],
                ),
                reverse=True,
            )
            return recommendations

        recommendations.sort(
            key=lambda item: (
                item["final_score"],
                item["knn_score"],
                item["matched_count"],
                -item["missing_count"],
                item["overlap_score"],
            ),
            reverse=True,
        )
        return recommendations


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    recommender = SmartKitchenRecommender(data_dir / "recipes_clean.json")
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
            print(f"   KNN similarity score: {recipe['knn_score']}")
            print(f"   Coverage score: {recipe['coverage_score']}")
            print(f"   Pantry overlap: {recipe['overlap_score']}")
            print(f"   Personalization bonus: {recipe['personalization_bonus']}")
            print(f"   Matched count: {recipe['matched_count']}")
            print(f"   Missing count: {recipe['missing_count']}")
            print(f"   Matched: {recipe['matched_ingredients']}")
            print(f"   Missing: {recipe['missing_ingredients']}")
            print(f"   Calories: {recipe['calories']}")
            print(f"   Why this recipe: {recipe['explanation']}")
            print()
