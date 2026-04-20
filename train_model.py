import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation import EVALUATION_QUERIES, is_relevant
from ml_model import FEATURE_NAMES, MODEL_PATH, extract_feature_vector, sigmoid
from recommender import SmartKitchenRecommender
from user_preferences import UserPreferenceStore


EPOCHS = 700
LEARNING_RATE = 0.08
L2_PENALTY = 0.001
MAX_EXTRA_QUERIES = 18
NEGATIVE_RATIO = 2


def build_training_queries(recommender: SmartKitchenRecommender) -> List[List[str]]:
    queries = [list(query) for query in EVALUATION_QUERIES]
    seen = {tuple(query) for query in queries}

    for recipe in recommender.recipes[::31]:
        ingredients = [
            item
            for item in recommender.normalize_ingredients(recipe.get("ingredients", []))
            if item not in recommender.pantry_staples
        ]
        if len(ingredients) < 3:
            continue

        query = sorted(ingredients)[:4]
        key = tuple(query)
        if key in seen:
            continue

        seen.add(key)
        queries.append(query)
        if len(queries) >= len(EVALUATION_QUERIES) + MAX_EXTRA_QUERIES:
            break

    return queries


def build_examples(
    recommender: SmartKitchenRecommender,
    profile: Dict,
) -> Tuple[List[List[float]], List[int]]:
    pantry_staples = recommender.resolve_pantry_staples(profile)
    positives = []
    negatives = []

    for query in build_training_queries(recommender):
        normalized_query = recommender.normalize_ingredients(query)
        query_positives = []
        query_negatives = []

        for index, recipe in enumerate(recommender.recipes):
            if not recommender.is_reasonable_recipe_name(recipe.get("name", "")):
                continue

            scored = recommender.score_recipe(
                recipe=recipe,
                normalized_user_ingredients=normalized_query,
                pantry_staples=pantry_staples,
                user_profile=profile,
            )
            if scored["calories"] is not None and scored["calories"] > 900:
                continue

            label = 1 if is_relevant(scored) else 0
            features = extract_feature_vector(scored)
            if label:
                query_positives.append(features)
            elif scored["matched_count"] > 0 or index % 193 == 0:
                query_negatives.append(features)

        positives.extend(query_positives)
        negatives.extend(query_negatives[: max(len(query_positives) * NEGATIVE_RATIO, 24)])

    examples = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    return examples, labels


def split_train_test(
    examples: List[List[float]],
    labels: List[int],
) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for index, (features, label) in enumerate(zip(examples, labels)):
        if index % 5 == 0:
            test_x.append(features)
            test_y.append(label)
        else:
            train_x.append(features)
            train_y.append(label)

    return train_x, train_y, test_x, test_y


def fit_scaler(examples: List[List[float]]) -> Tuple[List[float], List[float]]:
    feature_count = len(FEATURE_NAMES)
    means = []
    stds = []

    for column in range(feature_count):
        values = [row[column] for row in examples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance) or 1.0
        means.append(mean)
        stds.append(std)

    return means, stds


def scale_examples(
    examples: List[List[float]],
    means: List[float],
    stds: List[float],
) -> List[List[float]]:
    return [
        [(value - mean) / std for value, mean, std in zip(row, means, stds)]
        for row in examples
    ]


def train_logistic_regression(
    examples: List[List[float]],
    labels: List[int],
) -> Dict:
    train_x, train_y, test_x, test_y = split_train_test(examples, labels)
    means, stds = fit_scaler(train_x)
    scaled_train_x = scale_examples(train_x, means, stds)
    scaled_test_x = scale_examples(test_x, means, stds)

    weights = [0.0 for _ in FEATURE_NAMES]
    intercept = 0.0
    sample_count = len(scaled_train_x)

    for _ in range(EPOCHS):
        gradients = [0.0 for _ in FEATURE_NAMES]
        intercept_gradient = 0.0

        for features, label in zip(scaled_train_x, train_y):
            logit = intercept + sum(weight * value for weight, value in zip(weights, features))
            error = sigmoid(logit) - label
            intercept_gradient += error
            for index, value in enumerate(features):
                gradients[index] += error * value

        intercept -= LEARNING_RATE * intercept_gradient / sample_count
        for index in range(len(weights)):
            regularization = L2_PENALTY * weights[index]
            weights[index] -= LEARNING_RATE * ((gradients[index] / sample_count) + regularization)

    train_metrics = evaluate_model(scaled_train_x, train_y, weights, intercept)
    test_metrics = evaluate_model(scaled_test_x, test_y, weights, intercept)

    return {
        "model_type": "logistic_regression",
        "label_strategy": "pseudo_label_relevance",
        "feature_names": FEATURE_NAMES,
        "means": [round(value, 8) for value in means],
        "stds": [round(value, 8) for value in stds],
        "weights": [round(value, 8) for value in weights],
        "intercept": round(intercept, 8),
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "l2_penalty": L2_PENALTY,
            "train_examples": len(train_x),
            "test_examples": len(test_x),
            "positive_examples": sum(labels),
            "negative_examples": len(labels) - sum(labels),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        "feature_weights": build_feature_weights(weights),
    }


def evaluate_model(
    examples: List[List[float]],
    labels: List[int],
    weights: List[float],
    intercept: float,
) -> Dict:
    true_positive = false_positive = true_negative = false_negative = 0

    for features, label in zip(examples, labels):
        probability = sigmoid(intercept + sum(weight * value for weight, value in zip(weights, features)))
        prediction = 1 if probability >= 0.5 else 0

        if prediction == 1 and label == 1:
            true_positive += 1
        elif prediction == 1 and label == 0:
            false_positive += 1
        elif prediction == 0 and label == 0:
            true_negative += 1
        else:
            false_negative += 1

    total = max(len(labels), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "accuracy": round((true_positive + true_negative) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_feature_weights(weights: List[float]) -> List[Dict]:
    rows = [
        {
            "feature": feature,
            "weight": round(weight, 4),
            "direction": "positive" if weight > 0 else "negative" if weight < 0 else "neutral",
        }
        for feature, weight in zip(FEATURE_NAMES, weights)
    ]
    return sorted(rows, key=lambda item: abs(item["weight"]), reverse=True)


def save_model(model: Dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)


def main() -> None:
    recommender = SmartKitchenRecommender("recipes_clean.json")
    profile = UserPreferenceStore("user_profile.json").snapshot()
    examples, labels = build_examples(recommender, profile)

    if len(set(labels)) < 2:
        raise RuntimeError("Training data needs both positive and negative labels.")

    model = train_logistic_regression(examples, labels)
    save_model(model)

    print(f"Saved supervised ML ranker to {MODEL_PATH}")
    print(f"Examples: {len(labels)} ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    print(f"Test metrics: {model['training']['test_metrics']}")
    print("Top learned feature weights:")
    for row in model["feature_weights"][:8]:
        print(f"- {row['feature']}: {row['weight']:+.4f}")


if __name__ == "__main__":
    main()
