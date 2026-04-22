import sys
import os
import streamlit as st


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.nutrition import analyze, score
from src.features.grocery import generate
from src.preprocessing.data_cleaning import load_data
from src.kitchen_csp.csp_solver import heuristic_search

# Page Config
st.set_page_config(page_title="Smart AI Meal Planner", layout="centered")

st.title("🍽 Smart AI Meal Planner (Heuristic)")

# User Inputs
max_calories = st.slider("Max Calories", 1000, 3000, 2000)
min_protein = st.slider("Min Protein (g)", 0, 200, 50)
diet = st.selectbox("Select Diet", ["any", "vegetarian"])

# Load Data
recipes = load_data("data/processed/cleaned_recipes.csv")
# Generate Button
if st.button("Generate Meal Plan 🚀"):

    constraints = {
        "max_calories": max_calories,
        "min_protein": min_protein,
        "diet": diet,
    }

    meal_plan = heuristic_search(recipes, constraints)

    # Output
    if meal_plan:

        st.subheader("🍽 Meal Plan")
        for meal, recipe in meal_plan.items():
            st.write(f"{meal.capitalize()}: {recipe.name}")

        # Nutrition
        nutrition = analyze(meal_plan)

        st.subheader("🥗 Nutrition")
        st.write(f"Calories: {nutrition['calories']}")
        st.write(f"Protein: {nutrition['protein']}")

        # Score
        st.subheader("⭐ Plan Score")
        st.write(score(meal_plan, constraints))

        # Grocery list
        st.subheader("🛒 Grocery List")
        st.write(generate(meal_plan))

    else:
        st.error("❌ No solution found that meets the constraints.")
