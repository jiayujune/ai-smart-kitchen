import json
import os
from pathlib import Path
from typing import Dict, List
from urllib import error, request as urlrequest

from csp_solver import heuristic_search
from nutrition import score
from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

from evaluation import run_evaluation
from recommender import SmartKitchenRecommender
from user_preferences import UserPreferenceStore

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart-kitchen-dev-secret")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
recommender = SmartKitchenRecommender(DATA_DIR / "recipes_clean.json")
preferences = UserPreferenceStore(DATA_DIR / "user_profile.json")

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
    "ml_ranker": "ML Ranker",
    "csp": "CSP Meal Planner",
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
CSP_MEAL_ORDER = {"breakfast": 0, "lunch": 1, "dinner": 2}
DESSERT_KEYWORDS = {
    "cookie", "cookies", "brownie", "cake", "cupcake", "muffin", "frosting",
    "icing", "dessert", "chocolate", "peanut butter", "pie", "pudding",
    "sweet", "candy", "caramel"
}

MEAL_KEYWORDS = {
    "breakfast": {
        "bagel", "biscuit", "breakfast", "cereal", "egg", "eggs", "french toast",
        "granola", "muffin", "oatmeal", "oats", "omelet", "omelette", "pancake",
        "parfait", "sausage", "scramble", "smoothie", "toast", "waffle", "yogurt",
    },
    "lunch": {
        "bowl", "burger", "chicken", "lunch", "panini", "pasta", "quesadilla",
        "rice", "salad", "sandwich", "soup", "taco", "tuna", "turkey", "wrap",
    },
    "dinner": {
        "beef", "casserole", "curry", "dinner", "fish", "pasta", "pork", "rice",
        "roast", "salmon", "shrimp", "steak", "stew", "stir fry", "stir-fry",
        "taco", "tofu", "turkey", "vegetable", "veggie",
    },
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


class CSPRecipeAdapter:
    """Adapter so recipes from recommender.recipes work with csp_solver.py"""

    def __init__(self, recipe: Dict):
        nutrition = recipe.get("nutrition", [])
        self.recipe = recipe
        self.id = recipe.get("id")
        self.name = recipe.get("name", "unknown recipe")
        self.ingredients = recipe.get("ingredients", [])
        self.calories = self._safe_nutrition_value(nutrition, 0)
        self.protein = self._safe_nutrition_value(nutrition, 4)
        self.diet = "any"

    @staticmethod
    def _safe_nutrition_value(nutrition: List, index: int) -> float:
        if isinstance(nutrition, list) and len(nutrition) > index and isinstance(nutrition[index], (int, float)):
            return float(nutrition[index])
        return 0.0


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


def recipe_matches_dietary_filters(recipe: Dict, dietary_filters: List[str]) -> bool:
    if not dietary_filters:
        return True

    normalized_ingredients = recommender.normalize_ingredients(recipe.get("ingredients", []))
    ingredient_text = " | ".join(sorted(normalized_ingredients))

    for filter_name in dietary_filters:
        rule = recommender.dietary_rules.get(filter_name)
        if not rule:
            continue
        if any(term in ingredient_text for term in rule["blocked_terms"]):
            return False
    return True


def build_csp_constraints(dietary_filters: List[str]) -> Dict:
    diet = "vegetarian" if "vegetarian" in dietary_filters else "any"
    return {
        "max_calories": 2000,
        "min_protein": 50,
        "diet": diet,
    }


def recipe_matches_meal_slot(recipe: Dict, meal_type: str) -> bool:
    name_text = recommender.normalize_text(recipe.get("name", ""))
    ingredient_tokens = set(recommender.normalize_ingredients(recipe.get("ingredients", [])))
    haystack = f"{name_text} {' '.join(sorted(ingredient_tokens))}"

    dessert_words = {
        "cookie", "cookies", "brownie", "cake", "cupcake", "dessert",
        "chocolate", "peanut butter", "pie", "sweet", "candy", "caramel"
    }

    if any(word in haystack for word in dessert_words):
        return False

    breakfast_words = {
        "breakfast", "omelet", "omelette", "scramble", "toast",
        "pancake", "waffle", "oatmeal", "cereal", "smoothie",
        "granola", "parfait"
    }

    lunch_words = {
        "salad", "sandwich", "wrap", "burger", "quesadilla",
        "soup", "bowl", "lunch"
    }

    dinner_words = {
        "parmesan", "curry", "stir fry", "stir-fry", "roast",
        "steak", "casserole", "dinner"
    }

    if meal_type == "breakfast":
        return any(word in haystack for word in breakfast_words)

    if meal_type == "lunch":
        if any(word in haystack for word in lunch_words):
            return True
        return any(item in ingredient_tokens for item in {"bread", "lettuce", "rice", "tuna", "turkey"})

    if meal_type == "dinner":
        if any(word in haystack for word in dinner_words):
            return True
        return any(item in ingredient_tokens for item in {"chicken", "beef", "fish", "pasta", "rice", "tofu"})

    return False


def choose_best_csp_recipe(
    candidates: List[CSPRecipeAdapter],
    meal_type: str,
    normalized_items: List[str],
    pantry_staples,
    profile: Dict,
    dietary_filters: List[str],
):
    if not candidates:
        return None

    best_candidate = None
    best_score = float("-inf")
    normalized_set = set(normalized_items)

    for candidate in candidates:
        scored = recommender.score_recipe(
            recipe=candidate.recipe,
            normalized_user_ingredients=normalized_set,
            pantry_staples=pantry_staples,
            user_profile=profile,
            dietary_filters=dietary_filters,
        )

        current_score = float(scored.get("final_score") or 0)

        if current_score > best_score:
            best_score = current_score
            best_candidate = candidate

    return best_candidate

def build_csp_results(user_items: List[str], dietary_filters: List[str], profile: Dict) -> Dict:
    normalized_items = sorted(recommender.normalize_ingredients(user_items))
    pantry_staples = recommender.resolve_pantry_staples(profile)
    constraints = build_csp_constraints(dietary_filters)

    base_candidates = []
    for recipe in recommender.recipes:
        if not recommender.is_reasonable_recipe_name(recipe.get("name", "")):
            continue
        if not recipe_matches_dietary_filters(recipe, dietary_filters):
            continue

        adapter = CSPRecipeAdapter(recipe)
        if adapter.calories <= 0 or adapter.protein <= 0:
            continue

        scored = recommender.score_recipe(
            recipe=recipe,
            normalized_user_ingredients=set(normalized_items),
            pantry_staples=pantry_staples,
            user_profile=profile,
            dietary_filters=dietary_filters,
        )

        if normalized_items and scored["matched_count"] == 0:
            continue

        base_candidates.append(adapter)

    if not base_candidates:
        return {"results": [], "summary": None}

    selected_plan = {}
    used_recipe_ids = set()

    for meal_type in ["breakfast", "lunch", "dinner"]:
        meal_candidates = [
            adapter for adapter in base_candidates
            if adapter.id not in used_recipe_ids and recipe_matches_meal_slot(adapter.recipe, meal_type)
        ]

        # Do not force a bad match for this meal slot.
        if not meal_candidates:
           continue
        selected = choose_best_csp_recipe(
            candidates=meal_candidates,
            meal_type=meal_type,
            normalized_items=normalized_items,
            pantry_staples=pantry_staples,
            profile=profile,
            dietary_filters=dietary_filters,
        )

        if not selected:
            continue

        used_recipe_ids.add(selected.id)
        selected_plan[meal_type] = selected

    if not selected_plan:
        return {
            "results": [],
            "summary": None,
        }

    # Validate full-plan nutrition against the original CSP-style constraints.
    if score(selected_plan, constraints) == float("-inf"):
        return {
            "results": [],
            "summary": None,
        }

    results = []
    for meal_type in ["breakfast", "lunch", "dinner"]:
        selected = selected_plan.get(meal_type)
        if not selected:
            continue

        recipe = dict(selected.recipe)
        scored_recipe = recommender.score_recipe(
            recipe=recipe,
            normalized_user_ingredients=set(normalized_items),
            pantry_staples=pantry_staples,
            user_profile=profile,
            dietary_filters=dietary_filters,
        )
        scored_recipe["meal_slot"] = meal_type.title()
        scored_recipe["planner_type"] = "CSP"
        results.append(scored_recipe)

    totals = {
        "calories": round(sum(item.calories for item in selected_plan.values()), 1),
        "protein": round(sum(item.protein for item in selected_plan.values()), 1),
    }
    plan_score = round(score(selected_plan, constraints), 4)

    results.sort(key=lambda item: CSP_MEAL_ORDER.get(item.get("meal_slot", "").lower(), 99))

    summary = {
        "planner_type": "CSP",
        "score": plan_score,
        "total_calories": totals["calories"],
        "total_protein": totals["protein"],
        "constraints": constraints,
        "meal_count": len(results),
    }

    return {
        "results": results,
        "summary": summary,
    }


def build_chat_context(
    user_input: str,
    sort_mode: str,
    dietary_filters: List[str],
    results: List[Dict],
    profile: Dict,
    csp_summary: Dict = None,
) -> str:
    normalized_items = sorted(recommender.normalize_ingredients(parse_ingredients(user_input)))
    pantry_staples = profile.get("pantry_staples", [])
    active_dietary_labels = [DIETARY_FILTER_OPTIONS[item] for item in dietary_filters]
    top_results = []
    for item in results[:5]:
        substitution_parts = []
        for suggestion in item.get("substitution_suggestions", [])[:2]:
            option_names = [option["name"] for option in suggestion.get("options", [])[:2]]
            if option_names:
                substitution_parts.append(f"{suggestion['ingredient']} -> {', '.join(option_names)}")
        substitution_text = f" | substitutions={'; '.join(substitution_parts)}" if substitution_parts else ""
        meal_prefix = f"[{item.get('meal_slot')}] " if item.get("meal_slot") else ""
        top_results.append(
            f"- {meal_prefix}{item['name']} | match={item.get('match_percent', 0)}% | missing={item.get('missing_count', 0)} | calories={item.get('calories')}{substitution_text} | why={item.get('explanation', '')}"
        )

    result_block = "\n".join(top_results) if top_results else "- No strong results yet."
    csp_block = ""
    if csp_summary:
        csp_block = (
            f"\nCSP meal-plan summary: score={csp_summary['score']}, "
            f"total_calories={csp_summary['total_calories']}, "
            f"total_protein={csp_summary['total_protein']}.\n"
        )

    return (
        "You are a helpful Smart Kitchen assistant inside a recipe recommendation app. "
        "Only answer about meal ideas, recipe tradeoffs, pantry usage, substitutions, nutrition, "
        "and what the user can cook from their kitchen.\n\n"
        f"Current available ingredients: {', '.join(normalized_items) if normalized_items else 'None provided'}\n"
        f"Current pantry staples: {', '.join(pantry_staples) if pantry_staples else 'None configured'}\n"
        f"Current sort mode: {SORT_OPTIONS.get(sort_mode, 'Best Match')}\n"
        f"Current dietary filters: {', '.join(active_dietary_labels) if active_dietary_labels else 'None'}\n"
        f"{csp_block}"
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

    csp_summary = None
    if sort_mode == "csp":
        csp_data = build_csp_results(user_items, dietary_filters, profile)
        results = csp_data["results"]
        csp_summary = csp_data["summary"]
    else:
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
        if csp_summary:
            stats["planner_score"] = csp_summary["score"]
            stats["total_calories"] = csp_summary["total_calories"]
            stats["total_protein"] = csp_summary["total_protein"]

    return {
        "results": results,
        "normalized_items": normalized_items,
        "stats": stats,
        "profile": profile,
        "csp_summary": csp_summary,
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
        csp_summary=page_data["csp_summary"],
        example_ingredients=EXAMPLE_INGREDIENTS,
        quick_ingredients=quick_ingredients,
        recent_searches=recent_searches,
        recent_recipes=recent_recipes,
        pantry_staples=pantry_staples,
        staple_suggestions=staple_suggestions,
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
        csp_summary=page_data.get("csp_summary"),
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


@app.route("/evaluation")
def evaluation_dashboard():
    report = run_evaluation(
        recommender=recommender,
        profile=preferences.snapshot(),
    )
    return render_template("evaluation.html", report=report)


def build_saved_recipe_list(recipe_ids: List[int], profile: Dict) -> List[Dict]:
    liked_ids = set(profile.get("liked_recipes", []))
    favorite_ids = set(profile.get("favorites", []))
    items = []

    for recipe_id in recipe_ids:
        recipe = recommender.get_recipe_by_id(recipe_id)
        if not recipe:
            continue

        recipe_copy = dict(recipe)
        recipe_copy["calories"] = recommender.extract_calories(recipe.get("nutrition", []))
        recipe_copy["is_liked"] = recipe_id in liked_ids
        recipe_copy["is_favorite"] = recipe_id in favorite_ids
        recipe_copy["matched_ingredients"] = []
        recipe_copy["missing_ingredients"] = []
        recipe_copy["explanation"] = "Saved in your personal Smart Kitchen collection for quick access."
        items.append(recipe_copy)

    return items


@app.route("/my-recipes", methods=["GET", "POST"])
def my_recipes():
    if request.method == "POST":
        action = request.form.get("action", "").strip()
        recipe_id_raw = request.form.get("recipe_id", "").strip()
        if action in {"toggle_like", "toggle_favorite"} and recipe_id_raw.isdigit():
            recipe = recommender.get_recipe_by_id(int(recipe_id_raw))
            if recipe:
                if action == "toggle_like":
                    preferences.toggle_like(recipe)
                else:
                    preferences.toggle_favorite(recipe)
        return redirect(url_for("my_recipes"))

    profile = preferences.snapshot()
    liked_ids = profile.get("liked_recipes", [])
    favorite_ids = profile.get("favorites", [])

    return render_template(
        "my_recipes.html",
        liked_recipes=build_saved_recipe_list(liked_ids, profile),
        favorite_recipes=build_saved_recipe_list(favorite_ids, profile),
    )


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
    score_detail = None
    if normalized_items:
        score_detail = recommender.score_recipe(
            recipe=recipe,
            normalized_user_ingredients=set(normalized_items),
            pantry_staples=pantry_staples,
            user_profile=preferences.snapshot(),
            dietary_filters=active_dietary_filters,
        )

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
        score_detail=score_detail,
    )


if __name__ == "__main__":
    app.run(debug=True)
