def calorie_constraint(assignment, max_calories):
    total = sum(r.calories for r in assignment.values())
    return total <= max_calories


def diet_constraint(recipe, diet):
    return diet == "any" or recipe.diet == diet


def no_repeat_constraint(assignment):
    names = [r.name for r in assignment.values()]
    return len(names) == len(set(names))