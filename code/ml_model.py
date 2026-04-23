import json
import math
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "data" / "models" / "recipe_ranker.json"

FEATURE_NAMES = [
    "knn_score",
    "cosine_similarity",
    "jaccard_similarity",
    "coverage_score",
    "overlap_score",
    "matched_count",
    "missing_count",
    "total_ingredients",
    "missing_ratio",
    "matched_ratio",
    "calories_scaled",
    "calories_known",
    "calorie_bonus",
    "personalization_bonus",
]

FEATURE_LABELS = {
    "knn_score": "KNN similarity",
    "cosine_similarity": "Cosine similarity",
    "jaccard_similarity": "Jaccard similarity",
    "coverage_score": "Ingredient coverage",
    "overlap_score": "Pantry overlap",
    "matched_count": "Matched items",
    "missing_count": "Missing items",
    "total_ingredients": "Recipe size",
    "missing_ratio": "Missing ratio",
    "matched_ratio": "Matched ratio",
    "calories_scaled": "Calories",
    "calories_known": "Calories known",
    "calorie_bonus": "Calorie adjustment",
    "personalization_bonus": "Personalization",
}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def extract_feature_vector(candidate: Dict) -> List[float]:
    calories = candidate.get("calories")
    total_ingredients = max(float(candidate.get("total_ingredients") or 0), 1.0)
    matched_count = float(candidate.get("matched_count") or 0)
    missing_count = float(candidate.get("missing_count") or 0)

    values = {
        "knn_score": float(candidate.get("knn_score") or 0),
        "cosine_similarity": float(candidate.get("cosine_similarity") or 0),
        "jaccard_similarity": float(candidate.get("jaccard_similarity") or 0),
        "coverage_score": float(candidate.get("coverage_score") or 0),
        "overlap_score": float(candidate.get("overlap_score") or 0),
        "matched_count": matched_count,
        "missing_count": missing_count,
        "total_ingredients": total_ingredients,
        "missing_ratio": missing_count / total_ingredients,
        "matched_ratio": matched_count / total_ingredients,
        "calories_scaled": float(calories or 0) / 1000,
        "calories_known": 1.0 if calories is not None else 0.0,
        "calorie_bonus": float(candidate.get("calorie_bonus") or 0),
        "personalization_bonus": float(candidate.get("personalization_bonus") or 0),
    }
    return [values[name] for name in FEATURE_NAMES]


def load_ml_ranker(path: Path = MODEL_PATH) -> Optional[Dict]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        model = json.load(file)

    if model.get("feature_names") != FEATURE_NAMES:
        return None
    return model


def get_serving_model(model: Dict) -> Dict:
    if model.get("model_type") != "model_suite":
        return model

    default_model_type = model.get("default_model_type", "logistic_regression")
    for candidate in model.get("models", []):
        if candidate.get("model_type") == default_model_type:
            return candidate
    return model.get("models", [model])[0]


def predict_probability(candidate: Dict, model: Optional[Dict]) -> Optional[float]:
    if not model:
        return None

    serving_model = get_serving_model(model)
    features = extract_feature_vector(candidate)

    if serving_model.get("model_type") == "decision_tree":
        return round(predict_tree_probability(serving_model["tree"], features), 4)

    if serving_model.get("model_type") == "neural_network_mlp":
        scaled = scale_features(features, serving_model["means"], serving_model["stds"])
        return round(predict_mlp_probability(scaled, serving_model), 4)

    means = serving_model["means"]
    stds = serving_model["stds"]
    weights = serving_model["weights"]
    logit = float(serving_model["intercept"])

    for value, mean, std, weight in zip(features, means, stds, weights):
        logit += ((value - mean) / std) * weight

    return round(sigmoid(logit), 4)


def build_ml_contributions(candidate: Dict, model: Dict) -> List[Dict]:
    serving_model = get_serving_model(model)
    if serving_model.get("model_type") != "logistic_regression":
        return build_global_contributions(serving_model)

    features = extract_feature_vector(candidate)
    contributions = []

    for name, value, mean, std, weight in zip(
        FEATURE_NAMES,
        features,
        serving_model["means"],
        serving_model["stds"],
        serving_model["weights"],
    ):
        contribution = ((value - mean) / std) * weight
        contributions.append(
            {
                "label": FEATURE_LABELS[name],
                "value": round(contribution, 4),
                "raw_value": round(value, 4),
                "weight": round(weight, 4),
                "kind": "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral",
                "bar_width": min(100, round(abs(contribution) / 2.5 * 100)),
                "signed_value": f"{contribution:+.4f}",
            }
        )

    return sorted(contributions, key=lambda item: abs(item["value"]), reverse=True)[:8]


def scale_features(features: List[float], means: List[float], stds: List[float]) -> List[float]:
    return [(value - mean) / std for value, mean, std in zip(features, means, stds)]


def predict_tree_probability(tree: Dict, features: List[float]) -> float:
    node = tree
    while not node.get("leaf"):
        feature_index = node["feature_index"]
        node = node["left"] if features[feature_index] <= node["threshold"] else node["right"]
    return float(node["probability"])


def predict_mlp_probability(features: List[float], model: Dict) -> float:
    hidden_values = []
    for weights, hidden_bias in zip(model["hidden_weights"], model["hidden_biases"]):
        hidden_values.append(sigmoid(hidden_bias + sum(weight * value for weight, value in zip(weights, features))))
    return sigmoid(model["output_bias"] + sum(weight * value for weight, value in zip(model["output_weights"], hidden_values)))


def build_global_contributions(model: Dict) -> List[Dict]:
    rows = []
    for row in model.get("feature_weights", [])[:8]:
        value = float(row.get("weight", 0))
        rows.append(
            {
                "label": FEATURE_LABELS.get(row["feature"], row["feature"]),
                "value": round(value, 4),
                "raw_value": 0,
                "weight": round(value, 4),
                "kind": "positive" if value > 0 else "negative" if value < 0 else "neutral",
                "bar_width": min(100, round(abs(value) / 2.5 * 100)),
                "signed_value": f"{value:+.4f}",
            }
        )
    return rows


def attach_ml_prediction(candidate: Dict, model: Optional[Dict]) -> Dict:
    probability = predict_probability(candidate, model)
    candidate["ml_ranker_available"] = probability is not None
    candidate["ml_score"] = probability if probability is not None else 0.0
    candidate["ml_probability_percent"] = round((probability or 0.0) * 100)
    serving_model = get_serving_model(model) if model else {}
    candidate["ml_model_type"] = serving_model.get("model_type", "unavailable")
    candidate["ml_score_formula"] = "Supervised model probability learned from pseudo-labeled recipe relevance data."
    candidate["ml_contributions"] = build_ml_contributions(candidate, model) if model else []
    return candidate
