# AI Smart Kitchen

AI Smart Kitchen is a lightweight recipe recommendation prototype built with Python and Flask. It takes the ingredients already available in a user's kitchen, normalizes common ingredient variants, and returns recipe suggestions ranked by pantry fit, missing ingredients, and calorie profile.

## Project Goal

This project explores a practical smart-kitchen use case:

- reduce food waste by using ingredients already on hand
- help users discover recipes without starting from scratch
- provide simple nutrition-aware filtering for better everyday choices

## Current Features

- recipe data cleaning pipeline from `RAW_recipes.csv` to `recipes_clean.json`
- ingredient normalization for common pantry variations such as `eggs -> egg`
- rule-based ranking using:
  - recipe ingredient coverage
  - overlap with the user's pantry input
  - number of missing ingredients
  - basic calorie preference
- Flask web UI for interactive recipe search
- explanation text that tells the user why a recipe was recommended

## Tech Stack

- Python
- Flask
- Pandas
- HTML + inline CSS
- JSON recipe dataset

## Repository Structure

```text
.
|-- app.py                   # Flask entry point
|-- recommender.py           # Recommendation and ranking logic
|-- clean_recipes.py         # Dataset cleaning and preprocessing
|-- RAW_recipes.csv          # Raw source dataset
|-- recipes_clean.json       # Cleaned recipe subset
|-- templates/
|   `-- index.html           # Web interface
`-- AI_Recommendation_system.ipynb
```

## How It Works

### 1. Data Cleaning

`clean_recipes.py` loads the raw CSV and:

- keeps only useful columns such as recipe name, ingredients, nutrition, and steps
- safely parses stringified lists
- normalizes ingredient text
- removes invalid or empty rows
- filters recipes to a practical ingredient-count range
- exports a cleaned JSON file for fast loading in the app

### 2. Recommendation Logic

`recommender.py` performs the main ranking:

1. Normalize user ingredients and recipe ingredients
2. Compute ingredient coverage and pantry overlap
3. Filter out weak matches and extremely high-calorie recipes
4. Rank recipes by a combined score
5. Return matched ingredients, missing ingredients, calories, and explanation text

This is currently a rule-based recommendation engine rather than a trained machine learning model. That makes it easy to explain, debug, and improve for a class prototype or MVP.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install flask pandas
```

### 3. Run the web app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Optional: Regenerate the Cleaned Dataset

If you want to rebuild `recipes_clean.json` from the raw CSV:

```powershell
python clean_recipes.py
```

## Example Input

```text
egg, tomato, rice, onion
```

Example output behavior:

- recipes with more pantry overlap appear higher
- recipes needing fewer extra ingredients rank better
- lighter and balanced calorie ranges receive a small boost

## Limitations

- ingredient synonym coverage is still limited
- ranking is heuristic, not personalized
- only a subset of the full dataset is used for faster prototyping
- there is no user account, preference history, or allergy filtering yet

## Future Improvements

- add more ingredient synonym handling and fuzzy matching
- incorporate recipe steps into the interface
- support dietary filters such as vegetarian, low-calorie, or high-protein
- train a learned recommendation model using user behavior or embeddings
- deploy the app online for demo use

## License

This project includes an MIT `LICENSE` file in the repository.
