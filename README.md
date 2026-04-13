# AI Smart Kitchen

AI Smart Kitchen is a Flask-based pantry-first recipe recommender with a sci-fi HUD interface, interactive recipe controls, and an AI cooking assistant. The app helps users turn available ingredients into realistic meal options by ranking recipes against pantry fit, missing items, calorie profile, and lightweight preference history.

## Highlights

- Pantry-first recipe search using ingredient normalization and overlap scoring
- Multiple ranking modes: `Best Match`, `Fewest Missing`, `Lowest Calories`, and `Most Personalized`
- Smart pantry staples so common household items do not clutter missing-ingredient lists
- Like, save, and recent-search history to make recommendations feel more personal over time
- Recipe detail pages with matched ingredients, missing ingredients, nutrition, and steps
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
4. Applies calorie-aware and preference-aware adjustments
5. Returns ranked recipe cards with explanation text and drill-down details

## Interface Overview

The current web app is designed like a compact AI control console instead of a plain demo page.

- Top status bar for system identity and live-state presentation
- HUD-style search panel with dark glass surfaces and cyan/blue highlights
- Sortable recommendation cards with match, missing, calorie, and personalization signals
- Recent searches with per-item deletion
- Pantry staple management for everyday kitchen defaults
- AI assistant panel with:
  - standby state
  - suggested prompts inside the chat window
  - animated processing visuals while waiting for a reply
  - gradual response reveal for a more natural assistant feel

## Core Components

### `app.py`

Flask application entry point. Handles:

- recipe search page rendering
- sort mode selection
- likes, favorites, pantry staple actions, and recent-search deletion
- recipe detail routes
- AI assistant requests and chat reset behavior

### `recommender.py`

Implements the rule-based recommendation engine:

- recipe loading and lookup
- ingredient normalization and synonym handling
- pantry staple resolution
- match, overlap, and missing-ingredient computation
- final scoring and ranking helpers

### `user_preferences.py`

Stores lightweight user state in `user_profile.json`:

- favorites
- liked recipes
- ingredient frequency
- recent searches
- recipe interaction history
- pantry staples

### `clean_recipes.py`

Utility script for turning the raw recipe CSV into the cleaned JSON dataset used by the app.

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
|-- app.py
|-- recommender.py
|-- user_preferences.py
|-- clean_recipes.py
|-- recipes_clean.json
|-- RAW_recipes.csv
|-- user_profile.json
|-- templates/
|   |-- index.html
|   `-- recipe_detail.html
`-- AI_Recommendation_system.ipynb
```

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install flask pandas
```

### 3. Run the app

```powershell
python app.py
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
python app.py
```

Optional model override:

```powershell
$env:GROQ_CHAT_MODEL="llama-3.1-8b-instant"
```

## Optional: Rebuild The Cleaned Dataset

If you want to regenerate `recipes_clean.json` from `RAW_recipes.csv`:

```powershell
python clean_recipes.py
```

## Current Limitations

- Recommendation quality is still heuristic rather than learned
- Ingredient synonym handling is useful but not exhaustive
- No account system or multi-user state separation yet
- No allergy-safe filtering or hard dietary constraints yet
- The assistant depends on an external API key and network access

## Good Next Steps

- Add dietary filters such as vegetarian, high-protein, or low-sodium
- Add richer scoring explanations and visual progress indicators
- Support partial AJAX updates for like/save/delete actions
- Add screenshots or a short demo GIF to this README
- Deploy a hosted demo for portfolio or class presentation use

## License

This repository includes an MIT `LICENSE` file.
