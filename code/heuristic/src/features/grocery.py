def generate(meal_plan):
    grocery = {}

    for recipe in meal_plan.values():
        for item in recipe.ingredients:
            grocery[item] = grocery.get(item, 0) + 1

    return grocery