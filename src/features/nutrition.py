def analyze(meal_plan):
    total = {
        "calories": 0,
        "protein": 0
    }

    for meal, recipe in meal_plan.items():
        total["calories"] += recipe["calories"]
        total["protein"] += recipe["protein"]

    return total

def score(meal_plan, constraints):
    nutrition = analyze(meal_plan)

    score = 0

    if nutrition["calories"] <= constraints["max_calories"]:
        score += 50

    if nutrition["protein"] >= constraints.get("min_protein", 0):
        score += 50

    return score