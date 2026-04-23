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
DECISION_TREE_MAX_DEPTH = 5
DECISION_TREE_MIN_SAMPLES = 18
MLP_HIDDEN_UNITS = 8
MLP_EPOCHS = 260
MLP_LEARNING_RATE = 0.035
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


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


def train_decision_tree(
    examples: List[List[float]],
    labels: List[int],
) -> Dict:
    train_x, train_y, test_x, test_y = split_train_test(examples, labels)
    tree = build_tree(train_x, train_y, depth=0)
    train_probabilities = [predict_tree_probability(tree, row) for row in train_x]
    test_probabilities = [predict_tree_probability(tree, row) for row in test_x]

    return {
        "model_type": "decision_tree",
        "label_strategy": "pseudo_label_relevance",
        "feature_names": FEATURE_NAMES,
        "tree": tree,
        "training": {
            "max_depth": DECISION_TREE_MAX_DEPTH,
            "min_samples": DECISION_TREE_MIN_SAMPLES,
            "train_examples": len(train_x),
            "test_examples": len(test_x),
            "positive_examples": sum(labels),
            "negative_examples": len(labels) - sum(labels),
            "train_metrics": evaluate_probabilities(train_y, train_probabilities),
            "test_metrics": evaluate_probabilities(test_y, test_probabilities),
        },
        "feature_weights": build_tree_feature_weights(tree),
    }


def train_neural_network(
    examples: List[List[float]],
    labels: List[int],
) -> Dict:
    train_x, train_y, test_x, test_y = split_train_test(examples, labels)
    means, stds = fit_scaler(train_x)
    scaled_train_x = scale_examples(train_x, means, stds)
    scaled_test_x = scale_examples(test_x, means, stds)
    feature_count = len(FEATURE_NAMES)

    hidden_weights = [
        [math.sin((unit + 1) * (feature + 2)) * 0.08 for feature in range(feature_count)]
        for unit in range(MLP_HIDDEN_UNITS)
    ]
    hidden_biases = [0.0 for _ in range(MLP_HIDDEN_UNITS)]
    output_weights = [math.cos(unit + 1) * 0.08 for unit in range(MLP_HIDDEN_UNITS)]
    output_bias = 0.0

    for _ in range(MLP_EPOCHS):
        for features, label in zip(scaled_train_x, train_y):
            hidden_values = [
                sigmoid(hidden_biases[unit] + sum(weight * value for weight, value in zip(hidden_weights[unit], features)))
                for unit in range(MLP_HIDDEN_UNITS)
            ]
            output_logit = output_bias + sum(weight * value for weight, value in zip(output_weights, hidden_values))
            probability = sigmoid(output_logit)
            output_error = probability - label

            previous_output_weights = list(output_weights)
            for unit in range(MLP_HIDDEN_UNITS):
                output_weights[unit] -= MLP_LEARNING_RATE * (output_error * hidden_values[unit] + L2_PENALTY * output_weights[unit])
            output_bias -= MLP_LEARNING_RATE * output_error

            for unit in range(MLP_HIDDEN_UNITS):
                hidden_error = output_error * previous_output_weights[unit] * hidden_values[unit] * (1 - hidden_values[unit])
                for feature_index, value in enumerate(features):
                    hidden_weights[unit][feature_index] -= MLP_LEARNING_RATE * (
                        hidden_error * value + L2_PENALTY * hidden_weights[unit][feature_index]
                    )
                hidden_biases[unit] -= MLP_LEARNING_RATE * hidden_error

    train_probabilities = [
        predict_mlp_probability(row, hidden_weights, hidden_biases, output_weights, output_bias)
        for row in scaled_train_x
    ]
    test_probabilities = [
        predict_mlp_probability(row, hidden_weights, hidden_biases, output_weights, output_bias)
        for row in scaled_test_x
    ]

    return {
        "model_type": "neural_network_mlp",
        "label_strategy": "pseudo_label_relevance",
        "feature_names": FEATURE_NAMES,
        "means": [round(value, 8) for value in means],
        "stds": [round(value, 8) for value in stds],
        "hidden_units": MLP_HIDDEN_UNITS,
        "hidden_weights": [[round(value, 8) for value in row] for row in hidden_weights],
        "hidden_biases": [round(value, 8) for value in hidden_biases],
        "output_weights": [round(value, 8) for value in output_weights],
        "output_bias": round(output_bias, 8),
        "training": {
            "epochs": MLP_EPOCHS,
            "learning_rate": MLP_LEARNING_RATE,
            "l2_penalty": L2_PENALTY,
            "train_examples": len(train_x),
            "test_examples": len(test_x),
            "positive_examples": sum(labels),
            "negative_examples": len(labels) - sum(labels),
            "train_metrics": evaluate_probabilities(train_y, train_probabilities),
            "test_metrics": evaluate_probabilities(test_y, test_probabilities),
        },
        "feature_weights": build_mlp_feature_weights(hidden_weights, output_weights),
    }


def evaluate_model(
    examples: List[List[float]],
    labels: List[int],
    weights: List[float],
    intercept: float,
) -> Dict:
    probabilities = [
        sigmoid(intercept + sum(weight * value for weight, value in zip(weights, features)))
        for features in examples
    ]
    return evaluate_probabilities(labels, probabilities)


def evaluate_probabilities(labels: List[int], probabilities: List[float]) -> Dict:
    true_positive = false_positive = true_negative = false_negative = 0

    for probability, label in zip(probabilities, labels):
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


def gini(labels: List[int]) -> float:
    if not labels:
        return 0.0
    positive_rate = sum(labels) / len(labels)
    negative_rate = 1 - positive_rate
    return 1 - positive_rate**2 - negative_rate**2


def build_tree(examples: List[List[float]], labels: List[int], depth: int) -> Dict:
    probability = round(sum(labels) / max(len(labels), 1), 4)
    if (
        depth >= DECISION_TREE_MAX_DEPTH
        or len(examples) < DECISION_TREE_MIN_SAMPLES
        or len(set(labels)) == 1
    ):
        return {"leaf": True, "probability": probability, "samples": len(labels)}

    split = find_best_split(examples, labels)
    if not split:
        return {"leaf": True, "probability": probability, "samples": len(labels)}

    feature_index, threshold, gain = split
    left_examples = []
    left_labels = []
    right_examples = []
    right_labels = []
    for row, label in zip(examples, labels):
        if row[feature_index] <= threshold:
            left_examples.append(row)
            left_labels.append(label)
        else:
            right_examples.append(row)
            right_labels.append(label)

    if not left_examples or not right_examples:
        return {"leaf": True, "probability": probability, "samples": len(labels)}

    return {
        "leaf": False,
        "feature": FEATURE_NAMES[feature_index],
        "feature_index": feature_index,
        "threshold": round(threshold, 6),
        "gain": round(gain, 6),
        "probability": probability,
        "samples": len(labels),
        "left": build_tree(left_examples, left_labels, depth + 1),
        "right": build_tree(right_examples, right_labels, depth + 1),
    }


def find_best_split(examples: List[List[float]], labels: List[int]) -> Tuple[int, float, float]:
    base_impurity = gini(labels)
    best_gain = 0.0
    best_split = None

    for feature_index in range(len(FEATURE_NAMES)):
        values = sorted({row[feature_index] for row in examples})
        if len(values) <= 1:
            continue
        step = max(1, len(values) // 12)
        thresholds = [(values[index] + values[min(index + 1, len(values) - 1)]) / 2 for index in range(0, len(values) - 1, step)]

        for threshold in thresholds:
            left = [label for row, label in zip(examples, labels) if row[feature_index] <= threshold]
            right = [label for row, label in zip(examples, labels) if row[feature_index] > threshold]
            if len(left) < DECISION_TREE_MIN_SAMPLES or len(right) < DECISION_TREE_MIN_SAMPLES:
                continue
            weighted_impurity = (len(left) / len(labels)) * gini(left) + (len(right) / len(labels)) * gini(right)
            gain = base_impurity - weighted_impurity
            if gain > best_gain:
                best_gain = gain
                best_split = (feature_index, threshold, gain)

    return best_split


def predict_tree_probability(tree: Dict, features: List[float]) -> float:
    node = tree
    while not node.get("leaf"):
        feature_index = node["feature_index"]
        node = node["left"] if features[feature_index] <= node["threshold"] else node["right"]
    return float(node["probability"])


def predict_mlp_probability(
    features: List[float],
    hidden_weights: List[List[float]],
    hidden_biases: List[float],
    output_weights: List[float],
    output_bias: float,
) -> float:
    hidden_values = [
        sigmoid(hidden_bias + sum(weight * value for weight, value in zip(weights, features)))
        for weights, hidden_bias in zip(hidden_weights, hidden_biases)
    ]
    return sigmoid(output_bias + sum(weight * value for weight, value in zip(output_weights, hidden_values)))


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


def build_tree_feature_weights(tree: Dict) -> List[Dict]:
    gains = {feature: 0.0 for feature in FEATURE_NAMES}

    def visit(node: Dict) -> None:
        if node.get("leaf"):
            return
        gains[node["feature"]] += node.get("gain", 0.0) * node.get("samples", 1)
        visit(node["left"])
        visit(node["right"])

    visit(tree)
    rows = [
        {
            "feature": feature,
            "weight": round(value, 4),
            "direction": "positive" if value > 0 else "neutral",
        }
        for feature, value in gains.items()
        if value > 0
    ]
    return sorted(rows, key=lambda item: abs(item["weight"]), reverse=True)


def build_mlp_feature_weights(hidden_weights: List[List[float]], output_weights: List[float]) -> List[Dict]:
    rows = []
    for feature_index, feature in enumerate(FEATURE_NAMES):
        influence = sum(output_weights[unit] * hidden_weights[unit][feature_index] for unit in range(len(output_weights)))
        rows.append(
            {
                "feature": feature,
                "weight": round(influence, 4),
                "direction": "positive" if influence > 0 else "negative" if influence < 0 else "neutral",
            }
        )
    return sorted(rows, key=lambda item: abs(item["weight"]), reverse=True)


def build_model_suite(examples: List[List[float]], labels: List[int]) -> Dict:
    models = [
        train_logistic_regression(examples, labels),
        train_decision_tree(examples, labels),
        train_neural_network(examples, labels),
    ]
    default_model = max(models, key=lambda item: item["training"]["test_metrics"]["f1"])
    logistic_model = next(item for item in models if item["model_type"] == "logistic_regression")

    suite = {
        "model_type": "model_suite",
        "label_strategy": "pseudo_label_relevance",
        "default_model_type": default_model["model_type"],
        "feature_names": FEATURE_NAMES,
        "models": models,
        "model_comparison": [
            {
                "model_type": model["model_type"],
                "train_metrics": model["training"]["train_metrics"],
                "test_metrics": model["training"]["test_metrics"],
            }
            for model in models
        ],
        "training": default_model["training"],
        "feature_weights": default_model.get("feature_weights", []),
    }

    # Backward-compatible fields for the app's linear contribution display.
    suite["means"] = logistic_model["means"]
    suite["stds"] = logistic_model["stds"]
    suite["weights"] = logistic_model["weights"]
    suite["intercept"] = logistic_model["intercept"]
    return suite


def save_model(model: Dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)


def main() -> None:
    recommender = SmartKitchenRecommender(DATA_DIR / "recipes_clean.json")
    profile = UserPreferenceStore(DATA_DIR / "user_profile.json").snapshot()
    examples, labels = build_examples(recommender, profile)

    if len(set(labels)) < 2:
        raise RuntimeError("Training data needs both positive and negative labels.")

    model = build_model_suite(examples, labels)
    save_model(model)

    print(f"Saved supervised ML ranker to {MODEL_PATH}")
    print(f"Examples: {len(labels)} ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    print(f"Default model: {model['default_model_type']}")
    print("Model comparison:")
    for row in model["model_comparison"]:
        print(f"- {row['model_type']}: {row['test_metrics']}")
    print("Top learned feature weights from default model:")
    for row in model["feature_weights"][:8]:
        print(f"- {row['feature']}: {row['weight']:+.4f}")


if __name__ == "__main__":
    main()
