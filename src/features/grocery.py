def generate(meal_plan):
    grocery = {}

    for recipe in meal_plan.values():
        ingredients = recipe["ingredients"]

        for item in ingredients:
            grocery[item] = grocery.get(item, 0) + 1

    return grocery