import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.data_cleaning import load_data, detect_diet
from kitchen_csp.csp_solver import CSPSolver
from features.nutrition import analyze, score
from features.grocery import generate


st.set_page_config(page_title="Smart AI Meal Planner", layout="centered")

st.title("🍽 Smart AI Meal Planner (CSP)")

# User Inputs
max_calories = st.slider("Max Calories", 1000, 3000, 2000)
min_protein = st.slider("Min Protein (g)", 0, 200, 50)
diet = st.selectbox("Select Diet", ["any", "vegetarian"])


# Load Data
recipes = load_data("data/processed/cleaned_recipes.csv")

# diet column
for r in recipes:
    if "diet" not in r:
        r["diet"] = detect_diet(r["ingredients"])


# Solve using CSP
if st.button("Generate Meal Plan 🚀"):

    constraints = {
        "max_calories": max_calories,
        "min_protein": min_protein,
        "diet": diet
    }

    solver = CSPSolver(recipes, constraints)
    solution = solver.backtrack()

    if solution:
        st.subheader("🍽 Meal Plan")
        for meal, recipe in solution.items():
            st.write(f"{meal.capitalize()}: {recipe['name']}")

        nutrition = analyze(solution)

        st.subheader("🥗 Nutrition")
        st.write(f"Calories: {nutrition['calories']}")
        st.write(f"Protein: {nutrition['protein']}")

        st.subheader("⭐ Plan Score")
        st.write(score(solution, constraints))

        st.subheader("🛒 Grocery List")
        st.write(generate(solution))

    else:
        st.error("❌ No valid meal plan found.")
