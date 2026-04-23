def analyze(meal_plan):
    total = {"calories": 0, "protein": 0}

    for recipe in meal_plan.values():
        total["calories"] += recipe.calories
        total["protein"] += recipe.protein

    return total

def score(meal_plan, constraints):
    total_calories = sum(r.calories for r in meal_plan.values())
    total_protein = sum(r.protein for r in meal_plan.values())

    # Avoid invalid plans (calories and protein constraints)
    if total_calories > constraints["max_calories"]:
        return float('-inf')

    if total_protein < constraints["min_protein"]:
        return float('-inf')

    # Calculate variety score (more variety = higher score)
    main_ingredients = [ingredient for recipe in meal_plan.values() for ingredient in recipe.ingredients]
    unique_ingredients = set(main_ingredients)
    variety_score = len(unique_ingredients)

    # Prioritize high protein and variety within calorie limits
    return total_protein - total_calories / 15 + variety_score * 2