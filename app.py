from typing import Dict, List

from flask import Flask, abort, render_template, request

from recommender import SmartKitchenRecommender
from user_preferences import UserPreferenceStore

app = Flask(__name__)
recommender = SmartKitchenRecommender("recipes_clean.json")
preferences = UserPreferenceStore("user_profile.json")

DEFAULT_SETTINGS = {
    "top_k": 8,
    "min_score": 0.15,
    "min_matched_count": 2,
    "max_missing_count": 6,
}

EXAMPLE_INGREDIENTS = [
    "egg, tomato, rice, onion",
    "chicken breast, rice, garlic, onion",
    "cheese, tomato, onion, bell pepper",
]

DEFAULT_STAPLE_SUGGESTIONS = [
    "salt",
    "pepper",
    "oil",
    "water",
    "butter",
    "sugar",
    "soy sauce",
    "vinegar",
    "garlic",
]

SORT_OPTIONS = {
    "best_match": "Best Match",
    "fewest_missing": "Fewest Missing",
    "lowest_calories": "Lowest Calories",
    "most_personalized": "Most Personalized",
}

NUTRITION_LABELS = [
    ("Calories", 0),
    ("Total Fat", 1),
    ("Sugar", 2),
    ("Sodium", 3),
    ("Protein", 4),
    ("Saturated Fat", 5),
    ("Carbohydrates", 6),
]


def parse_ingredients(raw_text: str) -> List[str]:
    return [item.strip() for item in raw_text.split(",") if item.strip()]


def build_nutrition_items(nutrition: List) -> List[Dict]:
    items = []
    for label, index in NUTRITION_LABELS:
        value = nutrition[index] if isinstance(nutrition, list) and len(nutrition) > index else None
        items.append({"label": label, "value": value})
    return items


def build_results(user_input: str, sort_mode: str) -> Dict:
    user_items = parse_ingredients(user_input)
    normalized_items = sorted(recommender.normalize_ingredients(user_items))
    profile = preferences.snapshot()

    results = recommender.recommend_recipes(
        user_items,
        user_profile=profile,
        sort_mode=sort_mode,
        **DEFAULT_SETTINGS,
    )

    stats = None
    if normalized_items:
        stats = {
            "ingredients_count": len(normalized_items),
            "results_count": len(results),
            "best_match": results[0]["match_percent"] if results else 0,
            "liked_count": len(profile.get("liked_recipes", [])),
            "favorites_count": len(profile.get("favorites", [])),
            "staples_count": len(profile.get("pantry_staples", [])),
        }

    return {
        "results": results,
        "normalized_items": normalized_items,
        "stats": stats,
        "profile": profile,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    current_input = request.values.get("ingredients", "").strip()
    sort_mode = request.values.get("sort_mode", "best_match").strip() or "best_match"
    if sort_mode not in SORT_OPTIONS:
        sort_mode = "best_match"

    if request.method == "POST":
        action = request.form.get("action", "search")

        if action == "use_quick_ingredient":
            ingredient = request.form.get("ingredient", "").strip()
            current_items = parse_ingredients(current_input)
            normalized_current = recommender.normalize_ingredients(current_items)
            normalized_new = recommender.normalize_text(ingredient)
            if normalized_new and normalized_new not in normalized_current:
                current_items.append(ingredient)
            current_input = ", ".join(current_items)

        elif action in {"toggle_like", "toggle_favorite"}:
            recipe_id_raw = request.form.get("recipe_id", "").strip()
            if recipe_id_raw.isdigit():
                recipe = recommender.get_recipe_by_id(int(recipe_id_raw))
                if recipe:
                    if action == "toggle_like":
                        preferences.toggle_like(recipe)
                    else:
                        preferences.toggle_favorite(recipe)

        elif action == "add_staple":
            staple = recommender.normalize_text(request.form.get("staple", "").strip())
            if staple:
                preferences.add_pantry_staple(staple)

        elif action == "remove_staple":
            staple = recommender.normalize_text(request.form.get("staple", "").strip())
            if staple:
                preferences.remove_pantry_staple(staple)

        if current_input:
            normalized_for_store = sorted(recommender.normalize_ingredients(parse_ingredients(current_input)))
            preferences.record_search(normalized_for_store)

    page_data = build_results(current_input, sort_mode)
    profile = page_data["profile"]

    quick_ingredients = preferences.top_ingredients(limit=10)
    recent_searches = profile.get("search_history", [])[:5]
    recent_recipes = profile.get("recipe_history", [])[:6]
    pantry_staples = profile.get("pantry_staples", [])
    favorite_ids = set(profile.get("favorites", []))
    favorite_recipes = []
    for recipe_id in list(favorite_ids)[:6]:
        recipe = recommender.get_recipe_by_id(recipe_id)
        if recipe:
            favorite_recipes.append(recipe)

    staple_suggestions = [item for item in DEFAULT_STAPLE_SUGGESTIONS if item not in pantry_staples]

    return render_template(
        "index.html",
        results=page_data["results"],
        user_input=current_input,
        sort_mode=sort_mode,
        sort_options=SORT_OPTIONS,
        normalized_items=page_data["normalized_items"],
        stats=page_data["stats"],
        example_ingredients=EXAMPLE_INGREDIENTS,
        quick_ingredients=quick_ingredients,
        recent_searches=recent_searches,
        recent_recipes=recent_recipes,
        pantry_staples=pantry_staples,
        staple_suggestions=staple_suggestions,
        favorite_recipes=favorite_recipes,
    )


@app.route("/recipe/<int:recipe_id>")
def recipe_detail(recipe_id: int):
    recipe = recommender.get_recipe_by_id(recipe_id)
    if not recipe:
        abort(404)

    current_input = request.args.get("ingredients", "").strip()
    sort_mode = request.args.get("sort_mode", "best_match").strip() or "best_match"
    if sort_mode not in SORT_OPTIONS:
        sort_mode = "best_match"

    normalized_items = sorted(recommender.normalize_ingredients(parse_ingredients(current_input))) if current_input else []
    recipe_ingredients = sorted(recommender.normalize_ingredients(recipe.get("ingredients", [])))
    matched_items = sorted(set(normalized_items).intersection(recipe_ingredients))
    missing_items = sorted(item for item in recipe_ingredients if item not in set(normalized_items))
    pantry_staples = set(preferences.pantry_staples())
    missing_items = [item for item in missing_items if item not in pantry_staples]

    return render_template(
        "recipe_detail.html",
        recipe=recipe,
        user_input=current_input,
        sort_mode=sort_mode,
        sort_label=SORT_OPTIONS[sort_mode],
        normalized_items=normalized_items,
        matched_items=matched_items,
        missing_items=missing_items,
        nutrition_items=build_nutrition_items(recipe.get("nutrition", [])),
    )


if __name__ == "__main__":
    app.run(debug=True)
