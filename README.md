# AI Smart Kitchen

AI Smart Kitchen is a Flask-based pantry-first recipe recommender with a sci-fi HUD interface, interactive recipe controls, and an AI cooking assistant. The app helps users turn available ingredients into realistic meal options by ranking recipes against pantry fit, missing items, calorie profile, and lightweight preference history.

## Abstract

This project builds an AI-powered recipe recommendation system for pantry-first cooking. Given a user's available ingredients, the system cleans and normalizes recipe data, engineers ingredient-matching and nutrition features, ranks recipes through content-based similarity and supervised learning, and explains why each recommendation appears. The final web app includes dietary filters, substitution suggestions, user preference tracking, an AI cooking assistant, and an evaluation dashboard for comparing traditional, neural-network, and heuristic ranking methods.

## Developers

- Jiayu; Jhanavi; Amruta

## Highlights

- Pantry-first recipe search using ingredient normalization and overlap scoring
- Multiple ranking modes: `Best Match`, `Fewest Missing`, `Lowest Calories`, `Most Personalized`, and `ML Ranker`
- Dietary filters for vegetarian, gluten-free, vegan, dairy-free, and nut-free recipe searches
- Smart pantry staples so common household items do not clutter missing-ingredient lists
- Missing-ingredient substitution suggestions with pantry-aware highlights
- Like, save, and recent-search history to make recommendations feel more personal over time
- Dedicated `My Recipes` page for liked and saved recipes
- Recipe detail pages with matched ingredients, missing ingredients, nutrition, and steps
- Explainable AI score breakdowns for every recommendation
- Three supervised ML models trained from pseudo-labeled recipe relevance data
- Offline AI/ML evaluation dashboard comparing baseline, similarity, hybrid, traditional ML, and neural network rankers
- Built-in AI assistant for substitutions, meal triage, nutrition guidance, and quick cooking decisions
- HUD-inspired frontend with real-time system styling, animated assistant states, and progressive response reveal

## What The App Does

Given a comma-separated pantry input like:

```text
egg, tomato, rice, onion
```

the system:

1. Normalizes ingredient text into a cleaner comparison set
2. Scores recipes based on ingredient coverage and overlap
3. Penalizes recipes with too many missing ingredients
4. Applies selected dietary filters when needed
5. Suggests practical substitutes for missing ingredients
6. Applies calorie-aware and preference-aware adjustments
7. Optionally ranks candidates with the supervised ML model
8. Returns ranked recipe cards with explanation text and drill-down details

## AI Method

The recommender uses a hybrid, explainable ranking model. Each recipe is converted into a normalized ingredient set, then compared against the user's pantry input. The model combines:

- KNN-style ingredient similarity
- Cosine similarity
- Jaccard similarity
- Ingredient coverage
- Pantry overlap
- Missing-ingredient penalties
- Pantry-aware substitution suggestions
- Calorie-aware adjustments
- Lightweight personalization from likes, saves, views, and frequent ingredients

The final ranking score is:

```text
Final Score = 0.45 * KNN similarity
            + 0.25 * ingredient coverage
            + 0.10 * pantry overlap
            + 0.08 * matched ingredient count
            - 0.05 * missing ingredient count
            + calorie adjustment
            + personalization bonus
```

## Supervised ML Rankers

The project also includes a supervised machine learning model suite. It trains three pure-Python models on query-recipe pairs:

- Logistic Regression
- Decision Tree
- MLP Neural Network

- KNN, cosine, and Jaccard similarity
- Ingredient coverage and pantry overlap
- Matched and missing ingredient counts
- Missing and matched ratios
- Calories and calorie adjustment
- Personalization bonus

Because the dataset does not include explicit user ratings, the first version uses pseudo-labels:

```text
label = 1 when the recipe satisfies the relevance proxy
label = 0 otherwise
```

Train or refresh the supervised model:

```powershell
python code/train_model.py
```

The trained model is saved to:

```text
data/models/recipe_ranker.json
```

When that file exists, the app automatically enables the `ML Ranker` sorting mode and displays the predicted relevance probability on recipe cards. The app serves the trained model with the strongest test F1 score.

## Explainable AI

Each recipe card now shows an explainable score panel. The panel breaks the final score into individual positive and negative contributions, including similarity, coverage, matched ingredients, missing-ingredient penalty, calorie adjustment, and personalization. The recipe detail page reuses the same scoring logic so the explanation is consistent across the app.

This makes the recommender easier to defend in an AI course presentation because users can see not only what was recommended, but why it was ranked highly.

## Substitution Suggestions

For each recipe, the recommender checks missing ingredients against a curated substitution catalog. Suggested replacements are filtered through the active dietary rules and marked when the substitute is already present in the user's pantry input.

Examples:

- `butter` -> `oil`, `olive oil`, `applesauce`
- `chicken breast` -> `chicken thigh`, `turkey breast`, `tofu`
- `milk` -> `almond milk`, `soy milk`, `oat milk`, `water`

These suggestions make the app more useful in realistic cooking situations because a recipe with missing ingredients may still be practical if the user has a reasonable substitute.

## Offline Evaluation

The project includes `evaluation.py`, a local evaluation script that compares ranking strategies across representative pantry queries.

Evaluated rankers:

- Baseline matched ingredient count
- Jaccard similarity
- KNN similarity
- Hybrid explainable score
- Supervised ML ranker

Metrics:

- `Precision@8`: share of top recommendations that satisfy the relevance proxy
- Average missing ingredients
- Average match percentage
- Average final score

Run:

```powershell
python code/evaluation.py
```

You can also view the evaluation dashboard in the web app:

```text
http://127.0.0.1:5000/evaluation
```

Because the dataset does not include explicit user ratings, the script uses a transparent proxy label for relevance:

```text
matched_count >= 2
missing_count <= 6
coverage_score >= 0.25
knn_score >= 0.10
```

## Result Analysis

The model suite compares traditional and neural-network approaches on the same engineered recipe relevance features. Logistic Regression gives interpretable learned weights, Decision Tree gives non-linear rule splits, and the MLP Neural Network captures more flexible feature interactions. The evaluation dashboard reports train and test accuracy, precision, recall, and F1 for all three supervised models, making it easier to detect overfitting or underfitting.

Current offline ranking results show that the hybrid explainable ranker is strong for reducing missing ingredients, while the supervised ML ranker preserves high relevance using learned probabilities. The best model can vary as pseudo-labels and user feedback data change, so `train_model.py` stores all trained model metrics and serves the model with the strongest test F1.

The current train-test comparison shows stable generalization for Logistic Regression, the strongest held-out F1 for Decision Tree, and likely underfitting for the MLP Neural Network because its recall and F1 are lower than the traditional models. This is reported in the evaluation dashboard as a train-test gap and interpretation table.

Limitations remain: labels are pseudo-generated rather than collected from real users, evaluation queries are representative but small, and dietary/allergy logic is rule-based rather than medically certified. A natural next step is to collect real user interaction labels from clicks, likes, saves, and ignored recommendations, then retrain and compare the supervised models again.

## Interface Overview

The current web app is designed like a compact AI control console instead of a plain demo page.

- Top status bar for system identity and live-state presentation
- HUD-style search panel with dark glass surfaces and cyan/blue highlights
- Sortable recommendation cards with match, missing, calorie, and personalization signals
- Substitution panels for missing ingredients, including pantry-matched alternatives
- My Recipes vault for revisiting liked and saved recipes
- Recent searches with per-item deletion
- Pantry staple management for everyday kitchen defaults
- AI assistant panel with:
  - standby state
  - suggested prompts inside the chat window
  - animated processing visuals while waiting for a reply
  - gradual response reveal for a more natural assistant feel

## Core Components

### `code/app.py`

Flask application entry point. Handles:

- recipe search page rendering
- sort mode selection
- likes, favorites, pantry staple actions, and recent-search deletion
- recipe detail routes
- AI assistant requests and chat reset behavior

### `code/recommender.py`

Implements the rule-based recommendation engine:

- recipe loading and lookup
- ingredient normalization and synonym handling
- pantry staple resolution
- match, overlap, and missing-ingredient computation
- vegetarian, gluten-free, vegan, dairy-free, and nut-free hard filtering
- final scoring and ranking helpers

### `user_preferences.py`

Stores lightweight user state in `data/user_profile.json`:

- favorites
- liked recipes
- ingredient frequency
- recent searches
- recipe interaction history
- pantry staples

### `code/clean_recipes.py`

Utility script for turning the raw recipe CSV into the cleaned JSON dataset used by the app.

### `code/evaluation.py`

Offline evaluation script for comparing simple baselines, similarity rankers, the hybrid recommender, and the supervised ML ranker.

### `code/train_model.py`

Builds pseudo-labeled training data and trains Logistic Regression, Decision Tree, and MLP Neural Network rankers.

### `code/ml_model.py`

Loads the trained model, extracts feature vectors, predicts recipe relevance probability, and builds ML feature contribution explanations.

## Tech Stack

- Python
- Flask
- Pandas
- HTML/CSS/JavaScript
- JSON-based recipe data
- Groq-hosted chat completion API for the assistant

## Repository Structure

```text
.
|-- requirements.txt
|-- code/
|   |-- app.py
|   |-- recommender.py
|   |-- user_preferences.py
|   |-- ml_model.py
|   |-- train_model.py
|   |-- clean_recipes.py
|   |-- evaluation.py
|   |-- AI_Recommendation_system.ipynb
|   `-- templates/
|       |-- index.html
|       |-- evaluation.html
|       `-- recipe_detail.html
|-- data/
|   |-- RAW_recipes.csv
|   |-- recipes_clean.json
|   |-- user_profile.json
|   `-- models/
|       `-- recipe_ranker.json
|-- resources/
|   |-- README.md
|   `-- model_analysis.md
`-- README.md
```

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
python code/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## AI Assistant Setup

The in-app assistant uses a Groq API key.

You can provide it in either of two ways:

- Enter the key directly in the assistant panel inside the browser
- Set an environment variable before launching the app:

```powershell
$env:GROQ_API_KEY="your_key_here"
python code/app.py
```

Optional model override:

```powershell
$env:GROQ_CHAT_MODEL="llama-3.1-8b-instant"
```

## Optional: Rebuild The Cleaned Dataset

If you want to regenerate `data/recipes_clean.json` from `data/RAW_recipes.csv`:

```powershell
python code/clean_recipes.py
```

## Current Limitations

- The supervised model currently uses pseudo-labels rather than real explicit user ratings
- Ingredient synonym handling is useful but not exhaustive
- No account system or multi-user state separation yet
- Dietary filters are rule-based and not a certified allergy or medical safety system
- The assistant depends on an external API key and network access

## Good Next Steps

- Replace pseudo-labels with real feedback labels from likes, saves, clicks, and ignored recommendations
- Add more dietary modes such as high-protein, low-sodium, diabetic-friendly, or low-carb
- Support partial AJAX updates for like/save/delete actions
- Add screenshots or a short demo GIF to this README
- Deploy a hosted demo for portfolio or class presentation use

## License

This repository includes an MIT `LICENSE` file.
