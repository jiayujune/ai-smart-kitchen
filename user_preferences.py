import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class UserPreferenceStore:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self.load()

    def default_data(self) -> Dict:
        return {
            "favorites": [],
            "liked_recipes": [],
            "ingredient_counts": {},
            "search_history": [],
            "recipe_history": [],
            "recipe_feedback": {},
            "pantry_staples": [
                "salt",
                "pepper",
                "oil",
                "water",
                "flour",
                "sugar",
                "butter",
            ],
        }

    def load(self) -> Dict:
        if not self.storage_path.exists():
            data = self.default_data()
            self.save(data)
            return data

        with self.storage_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save(self, data: Dict = None) -> None:
        if data is not None:
            self.data = data

        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2, ensure_ascii=False)

    def snapshot(self) -> Dict:
        return json.loads(json.dumps(self.data))

    def _append_unique_recent(self, items: List[Dict], entry: Dict, key: str, limit: int) -> List[Dict]:
        filtered = [item for item in items if item.get(key) != entry.get(key)]
        filtered.insert(0, entry)
        return filtered[:limit]

    def record_search(self, ingredients: List[str]) -> None:
        cleaned = [item for item in ingredients if item]
        if not cleaned:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        joined = ", ".join(cleaned)

        for ingredient in cleaned:
            self.data["ingredient_counts"][ingredient] = self.data["ingredient_counts"].get(ingredient, 0) + 1

        self.data["search_history"] = self._append_unique_recent(
            self.data["search_history"],
            {
                "ingredients": cleaned,
                "label": joined,
                "timestamp": timestamp,
            },
            key="label",
            limit=8,
        )
        self.save()

    def update_recipe_feedback(self, recipe_id: int, field: str, delta: int) -> None:
        recipe_key = str(recipe_id)
        stats = self.data["recipe_feedback"].setdefault(recipe_key, {"likes": 0, "favorites": 0, "views": 0})
        stats[field] = max(0, stats.get(field, 0) + delta)

    def record_recipe_event(self, recipe: Dict, event_type: str) -> None:
        recipe_id = recipe.get("id")
        if recipe_id is None:
            return

        recipe_key = str(recipe_id)
        stats = self.data["recipe_feedback"].setdefault(recipe_key, {"likes": 0, "favorites": 0, "views": 0})
        stats["views"] = stats.get("views", 0) + 1

        entry = {
            "id": recipe_id,
            "name": recipe.get("name", "Unknown recipe"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "event": event_type,
        }
        self.data["recipe_history"] = self._append_unique_recent(
            self.data["recipe_history"],
            entry,
            key="id",
            limit=10,
        )
        self.save()

    def toggle_like(self, recipe: Dict) -> None:
        recipe_id = recipe.get("id")
        if recipe_id is None:
            return

        liked = self.data["liked_recipes"]
        if recipe_id in liked:
            liked.remove(recipe_id)
            self.update_recipe_feedback(recipe_id, "likes", -1)
        else:
            liked.insert(0, recipe_id)
            self.update_recipe_feedback(recipe_id, "likes", 1)
            self.record_recipe_event(recipe, "liked")
        self.save()

    def toggle_favorite(self, recipe: Dict) -> None:
        recipe_id = recipe.get("id")
        if recipe_id is None:
            return

        favorites = self.data["favorites"]
        if recipe_id in favorites:
            favorites.remove(recipe_id)
            self.update_recipe_feedback(recipe_id, "favorites", -1)
        else:
            favorites.insert(0, recipe_id)
            self.update_recipe_feedback(recipe_id, "favorites", 1)
            self.record_recipe_event(recipe, "favorited")
        self.save()

    def top_ingredients(self, limit: int = 10) -> List[str]:
        counts = Counter(self.data.get("ingredient_counts", {}))
        return [ingredient for ingredient, _ in counts.most_common(limit)]

    def pantry_staples(self) -> List[str]:
        return list(self.data.get("pantry_staples", []))

    def add_pantry_staple(self, ingredient: str) -> None:
        cleaned = ingredient.strip().lower()
        if not cleaned:
            return

        staples = self.data.setdefault("pantry_staples", [])
        if cleaned not in staples:
            staples.append(cleaned)
            staples.sort()
            self.save()

    def remove_pantry_staple(self, ingredient: str) -> None:
        cleaned = ingredient.strip().lower()
        staples = self.data.setdefault("pantry_staples", [])
        if cleaned in staples:
            staples.remove(cleaned)
            self.save()
