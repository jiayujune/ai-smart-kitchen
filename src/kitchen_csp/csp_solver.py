class CSPSolver:
    def __init__(self, recipes, constraints):
        self.recipes = recipes
        self.constraints = constraints
        self.variables = ["breakfast", "lunch", "dinner"]

    def is_valid(self, assignment):
        total_calories = sum(r["calories"] for r in assignment.values())
        total_protein = sum(r.get("protein", 0) for r in assignment.values())

        if total_calories > self.constraints["max_calories"]:
            return False

        if total_protein < self.constraints.get("min_protein", 0):
            return False
        for recipe in assignment.values():
            if (
                self.constraints["diet"] != "any"
                and recipe["diet"] != self.constraints["diet"]
            ):
                return False

        names = [r["name"] for r in assignment.values()]
        if len(names) != len(set(names)):
            return False

        return True

    def select_unassigned_variable(self, assignment):
        for v in self.variables:
            if v not in assignment:
                return v

    def backtrack(self, assignment=None):
        if assignment is None:
            assignment = {}

        if len(assignment) == len(self.variables):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for recipe in self.recipes:
            assignment[var] = recipe

            if self.is_valid(assignment):
                result = self.backtrack(assignment)
                if result:
                    return result

            del assignment[var]

        return None
