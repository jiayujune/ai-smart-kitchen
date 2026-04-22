from src.features.nutrition import score
from random import sample

import time
def heuristic_search(recipes, constraints):
    start_time = time.time()
    best_plan = None
    best_score = float('-inf')

    # Pre-filter recipes oncec before starting the loop
    valid_recipes = [r for r in recipes if
                     diet_constraint(r, constraints["diet"]) and r.calories > 0 and r.protein > 0]

    print(f"Filtering recipes took {time.time() - start_time:.4f} seconds.")

    for _ in range(100):  # iterations high for diversity
        plan = {}
        selected_recipes = set()

        for meal_type in ['breakfast', 'lunch', 'dinner']:
            # Randomly sample from the pre-filtered list
            meal_recipe = sample(valid_recipes, 1)[0]

            # Ensure the meal is unique
            while meal_recipe.name in selected_recipes:
                meal_recipe = sample(valid_recipes, 1)[0]

            plan[meal_type] = meal_recipe
            selected_recipes.add(meal_recipe.name)

        # Ensure variety is considered
        if variety_constraint(plan):
            current_score = score(plan, constraints)
            if current_score > best_score:
                best_score = current_score
                best_plan = plan

    print(f"Search took {time.time() - start_time:.4f} seconds.")
    return best_plan


def diet_constraint(recipe, diet):
    if diet == "any":
        return True
    if diet == "vegetarian":
        # Define a list of non-vegetarian ingredients
        non_vegetarian_ingredients = ["beef", "chicken", "pork", "fish", "tuna", "ham", "bacon", "lamb", "duck",
                                      "turkey", "shrimp", "crab", "salmon"]

        # Check if the recipe contains any non-vegetarian ingredients
        for ingredient in recipe.ingredients:
            if any(non_veg in ingredient.lower() for non_veg in non_vegetarian_ingredients):
                return False  # Exclude recipe if it contains non-vegetarian ingredients

        return True
    return recipe.diet == diet

def variety_constraint(assignment):
    if len(assignment) < 2:
        return True

    main_ingredients = [r.ingredients[0] for r in assignment.values()]
    return len(set(main_ingredients)) >= 2  # Ensure at least 2 different ingredients
