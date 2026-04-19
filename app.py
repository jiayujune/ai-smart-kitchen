import json
import os
from typing import Dict, List
from urllib import error, request as urlrequest

from flask import Flask, abort, jsonify, render_template, request, session

from recommender import SmartKitchenRecommender
from user_preferences import UserPreferenceStore

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-kitchen-dev-secret")
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

DIETARY_FILTER_OPTIONS = {
    "vegetarian": "Vegetarian",
    "gluten_free": "Gluten Free",
    "vegan": "Vegan",
    "dairy_free": "Dairy Free",
    "nut_free": "Nut Free",
}

CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
CHAT_HISTORY_LIMIT = 8

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


def parse_dietary_filters(raw_values: List[str]) -> List[str]:
    return [item for item in raw_values if item in DIETARY_FILTER_OPTIONS]


def build_nutrition_items(nutrition: List) -> List[Dict]:
    items = []
    for label, index in NUTRITION_LABELS:
        value = nutrition[index] if isinstance(nutrition, list) and len(nutrition) > index else None
        items.append({"label": label, "value": value})
    return items


def build_chat_context(
    user_input: str,
    sort_mode: str,
    dietary_filters: List[str],
    results: List[Dict],
    profile: Dict,
) -> str:
    normalized_items = sorted(recommender.normalize_ingredients(parse_ingredients(user_input)))
    pantry_staples = profile.get("pantry_staples", [])
    active_dietary_labels = [DIETARY_FILTER_OPTIONS[item] for item in dietary_filters]
    top_results = []
    for item in results[:5]:
        top_results.append(
            f"- {item['name']} | match={item['match_percent']}% | missing={item['missing_count']} | calories={item['calories']} | why={item['explanation']}"
        )

    result_block = "\n".join(top_results) if top_results else "- No strong results yet."
    return (
        "You are a helpful Smart Kitchen assistant inside a recipe recommendation app. "
        "Only answer about meal ideas, recipe tradeoffs, pantry usage, substitutions, nutrition, "
        "and what the user can cook from their kitchen.\n\n"
        f"Current available ingredients: {', '.join(normalized_items) if normalized_items else 'None provided'}\n"
        f"Current pantry staples: {', '.join(pantry_staples) if pantry_staples else 'None configured'}\n"
        f"Current sort mode: {SORT_OPTIONS.get(sort_mode, 'Best Match')}\n"
        f"Current dietary filters: {', '.join(active_dietary_labels) if active_dietary_labels else 'None'}\n"
        f"Current top recommendations:\n{result_block}\n\n"
        "When possible, reference the recommendations already on screen. Keep answers concise, practical, and cooking-focused."
    )


def call_openai_chat(messages: List[Dict], api_key_override: str = "") -> str:
    api_key = api_key_override.strip() or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing Groq API key. Add one in the Kitchen Assistant or set GROQ_API_KEY.")

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    req = urlrequest.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SmartKitchenAssistant/1.0",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq API error: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError("Unable to reach the Groq API from this environment.") from exc

    return data["choices"][0]["message"]["content"].strip()


def build_results(user_input: str, sort_mode: str, dietary_filters: List[str]) -> Dict:
    user_items = parse_ingredients(user_input)
    normalized_items = sorted(recommender.normalize_ingredients(user_items))
    profile = preferences.snapshot()

    results = recommender.recommend_recipes(
        user_items,
        user_profile=profile,
        sort_mode=sort_mode,
        dietary_filters=dietary_filters,
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
            "dietary_filter_count": len(dietary_filters),
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
    active_dietary_filters = parse_dietary_filters(request.values.getlist("dietary_filters"))
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

        elif action == "remove_search":
            label = request.form.get("search_label", "").strip()
            if label:
                preferences.remove_search(label)

        if current_input:
            normalized_for_store = sorted(recommender.normalize_ingredients(parse_ingredients(current_input)))
            preferences.record_search(normalized_for_store)

    page_data = build_results(current_input, sort_mode, active_dietary_filters)
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
        dietary_filter_options=DIETARY_FILTER_OPTIONS,
        active_dietary_filters=active_dietary_filters,
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
@app.route("/my-recipes")
def my_recipes():
    profile = preferences.snapshot()

    liked_ids = profile.get("liked_recipes", [])
    favorite_ids = profile.get("favorites", [])

    def build_recipe_list(recipe_ids):
        items = []
        for recipe_id in recipe_ids:
            recipe = recommender.get_recipe_by_id(recipe_id)
            if recipe:
                recipe_copy = dict(recipe)
                recipe_copy["calories"] = recommender.extract_calories(recipe.get("nutrition", []))
                recipe_copy["is_liked"] = recipe_id in liked_ids
                recipe_copy["is_favorite"] = recipe_id in favorite_ids
                recipe_copy["matched_ingredients"] = []
                recipe_copy["missing_ingredients"] = []
                recipe_copy["personalization_bonus"] = 0
                recipe_copy["explanation"] = "Saved in your personal Smart Kitchen collection for quick access."
                items.append(recipe_copy)
        return items

    liked_recipes = build_recipe_list(liked_ids)
    favorite_recipes = build_recipe_list(favorite_ids)

    return render_template(
        "my_recipes.html",
        liked_recipes=liked_recipes,
        favorite_recipes=favorite_recipes,
    )
@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    user_input = (payload.get("ingredients") or "").strip()
    api_key_override = (payload.get("api_key") or "").strip()
    sort_mode = (payload.get("sort_mode") or "best_match").strip()
    dietary_filters = parse_dietary_filters(payload.get("dietary_filters") or [])
    if sort_mode not in SORT_OPTIONS:
        sort_mode = "best_match"

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    page_data = build_results(user_input, sort_mode, dietary_filters)
    profile = page_data["profile"]
    chat_history = session.get("chat_history", [])

    system_prompt = build_chat_context(
        user_input=user_input,
        sort_mode=sort_mode,
        dietary_filters=dietary_filters,
        results=page_data["results"],
        profile=profile,
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-CHAT_HISTORY_LIMIT:])
    messages.append({"role": "user", "content": user_message})

    try:
        assistant_text = call_openai_chat(messages, api_key_override=api_key_override)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    chat_history.extend(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    session["chat_history"] = chat_history[-CHAT_HISTORY_LIMIT:]

    return jsonify({"reply": assistant_text})


@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    session.pop("chat_history", None)
    return jsonify({"ok": True})


@app.route("/recipe/<int:recipe_id>")
def recipe_detail(recipe_id: int):
    recipe = recommender.get_recipe_by_id(recipe_id)
    if not recipe:
        abort(404)

    current_input = request.args.get("ingredients", "").strip()
    sort_mode = request.args.get("sort_mode", "best_match").strip() or "best_match"
    active_dietary_filters = parse_dietary_filters(request.args.getlist("dietary_filters"))
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
        dietary_filters=active_dietary_filters,
        normalized_items=normalized_items,
        matched_items=matched_items,
        missing_items=missing_items,
        nutrition_items=build_nutrition_items(recipe.get("nutrition", [])),
    )


if __name__ == "__main__":
    app.run(debug=True)
