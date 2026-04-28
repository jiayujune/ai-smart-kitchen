class Recipe:
    def __init__(self, name, calories, protein, ingredients, diet, meal_type):
        self.name = name
        self.calories = calories
        self.protein = protein
        self.ingredients = ingredients
        self.diet = diet
        self.meal_type = meal_type

    @staticmethod
    def from_dict(data):
        return Recipe(
            name=data.get("name", ""),
            calories=data.get("calories", 0),
            protein=data.get("protein", 0),
            ingredients=data.get("ingredients", []),
            diet=data.get("diet", "any"),
            meal_type=data.get("meal_type", "dinner")
        )