def calorie_constraint(assignment, max_calories):
    total = sum(r.calories for r in assignment.values())
    return total <= max_calories


def diet_constraint(recipe, diet):
    return diet == "any" or recipe.diet == diet


def no_repeat_constraint(assignment):
    names = [r.name for r in assignment.values()]
    return len(names) == len(set(names))


def meal_type_constraint(meal, recipe):
    return recipe.meal_type == meal


def variety_constraint(assignment):
    if len(assignment) < 2:
        return True

    main_ingredients = [r.ingredients[0] for r in assignment.values()]
    return len(set(main_ingredients)) >= 2