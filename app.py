from typing import Dict, List

from flask import Flask, render_template, request

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


def parse_ingredients(raw_text: str) -> List[str]:
    return [item.strip() for item in raw_text.split(",") if item.strip()]


def build_results(user_input: str) -> Dict:
    user_items = parse_ingredients(user_input)
    normalized_items = sorted(recommender.normalize_ingredients(user_items))
    profile = preferences.snapshot()

    results = recommender.recommend_recipes(
        user_items,
        user_profile=profile,
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

        if current_input:
            normalized_for_store = sorted(recommender.normalize_ingredients(parse_ingredients(current_input)))
            preferences.record_search(normalized_for_store)

    page_data = build_results(current_input)
    profile = page_data["profile"]

    quick_ingredients = preferences.top_ingredients(limit=10)
    recent_searches = profile.get("search_history", [])[:5]
    recent_recipes = profile.get("recipe_history", [])[:6]
    favorite_ids = set(profile.get("favorites", []))
    favorite_recipes = []
    for recipe_id in list(favorite_ids)[:6]:
        recipe = recommender.get_recipe_by_id(recipe_id)
        if recipe:
            favorite_recipes.append(recipe)

    return render_template(
        "index.html",
        results=page_data["results"],
        user_input=current_input,
        normalized_items=page_data["normalized_items"],
        stats=page_data["stats"],
        example_ingredients=EXAMPLE_INGREDIENTS,
        quick_ingredients=quick_ingredients,
        recent_searches=recent_searches,
        recent_recipes=recent_recipes,
        favorite_recipes=favorite_recipes,
    )


if __name__ == "__main__":
    app.run(debug=True)
