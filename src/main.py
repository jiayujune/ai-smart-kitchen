from src.features.grocery import generate
from src.preprocessing.data_cleaning import load_data
from src.kitchen_csp.csp_solver import heuristic_search
import streamlit as st

def main():
    # Streamlit interface
    st.title("Smart AI Meal Planner")

    # Input sliders for user goals
    max_calories = st.slider('Max Calories', min_value=1000, max_value=3000, value=2000)
    min_protein = st.slider('Min Protein (g)', min_value=0, max_value=200, value=50)

    # Show user inputs
    st.write(f"Max Calories: {max_calories}")
    st.write(f"Min Protein: {min_protein}")

    # Load recipes and apply constraints
    recipes = load_data('data/processed/cleaned_recipes.csv')
    constraints = {
        "max_calories": max_calories,
        "min_protein": min_protein
    }

    # Apply heuristic search to get the best meal plan
    meal_plan, total_calories, total_protein = heuristic_search(recipes, constraints)

    if meal_plan:
        st.write("🍽 Meal Plan:")
        for meal, recipe in meal_plan.items():
            st.write(f"{meal.capitalize()}: {recipe.name}")

        st.write("\n🥗 Nutrition:")
        st.write(f"Total Calories: {total_calories}")
        st.write(f"Total Protein: {total_protein}")

        st.write("\n🛒 Grocery List:")
        st.write(generate(meal_plan))
    else:
        st.write("❌ No solution found that meets the constraints.")

if __name__ == "__main__":
    main()