from statistics import mean
from pathlib import Path
from typing import Callable, Dict, List

from recommender import SmartKitchenRecommender
from user_preferences import UserPreferenceStore


TOP_K = 8
MIN_SCORE = 0.15
MIN_MATCHED_COUNT = 2
MAX_MISSING_COUNT = 6
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

EVALUATION_QUERIES = [
    ["egg", "tomato", "rice", "onion"],
    ["chicken breast", "rice", "garlic", "onion"],
    ["cheese", "tomato", "onion", "bell pepper"],
    ["pasta", "garlic", "butter", "parmesan cheese"],
    ["beef", "potato", "carrot", "onion"],
    ["banana", "oats", "milk", "peanut butter"],
]


def is_relevant(candidate: Dict) -> bool:
    """Proxy label for offline evaluation when explicit user ratings are unavailable."""
    return (
        candidate["matched_count"] >= 2
        and candidate["missing_count"] <= 6
        and candidate["coverage_score"] >= 0.25
        and candidate["knn_score"] >= 0.10
    )


def build_candidates(
    recommender: SmartKitchenRecommender,
    profile: Dict,
    query: List[str],
) -> List[Dict]:
    normalized_query = recommender.normalize_ingredients(query)
    pantry_staples = recommender.resolve_pantry_staples(profile)
    candidates = []

    for recipe in recommender.recipes:
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
        if (
            scored["knn_score"] < MIN_SCORE
            or scored["matched_count"] < MIN_MATCHED_COUNT
            or scored["missing_count"] > MAX_MISSING_COUNT
        ):
            continue

        scored["is_relevant"] = is_relevant(scored)
        candidates.append(scored)

    return candidates


def rank_candidates(
    candidates: List[Dict],
    ranker: Callable[[Dict], tuple],
) -> List[Dict]:
    return sorted(candidates, key=ranker, reverse=True)


def evaluate_ranker(name: str, ranked: List[Dict], top_k: int = TOP_K) -> Dict:
    top_items = ranked[:top_k]
    if not top_items:
        return {
            "ranker": name,
            "precision_at_k": 0.0,
            "avg_missing": 0.0,
            "avg_match_percent": 0.0,
            "avg_final_score": 0.0,
        }

    return {
        "ranker": name,
        "precision_at_k": round(sum(item["is_relevant"] for item in top_items) / len(top_items), 3),
        "avg_missing": round(mean(item["missing_count"] for item in top_items), 2),
        "avg_match_percent": round(mean(item["match_percent"] for item in top_items), 2),
        "avg_final_score": round(mean(item["final_score"] for item in top_items), 4),
    }


def aggregate(results: List[Dict]) -> Dict:
    return {
        "ranker": results[0]["ranker"],
        "precision_at_k": round(mean(item["precision_at_k"] for item in results), 3),
        "avg_missing": round(mean(item["avg_missing"] for item in results), 2),
        "avg_match_percent": round(mean(item["avg_match_percent"] for item in results), 2),
        "avg_final_score": round(mean(item["avg_final_score"] for item in results), 4),
    }


def print_table(rows: List[Dict]) -> None:
    headers = ["Ranker", "Precision@8", "Avg Missing", "Avg Match %", "Avg Final Score"]
    widths = [22, 12, 12, 13, 16]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        values = [
            row["ranker"],
            f"{row['precision_at_k']:.3f}",
            f"{row['avg_missing']:.2f}",
            f"{row['avg_match_percent']:.2f}",
            f"{row['avg_final_score']:.4f}",
        ]
        print(" | ".join(value.ljust(width) for value, width in zip(values, widths)))


def get_rankers(include_ml_ranker: bool = False) -> Dict[str, Callable[[Dict], tuple]]:
    rankers = {
        "Baseline matched count": lambda item: (
            item["matched_count"],
            -item["missing_count"],
            item["match_percent"],
        ),
        "Jaccard similarity": lambda item: (
            item["jaccard_similarity"],
            item["matched_count"],
            -item["missing_count"],
        ),
        "KNN similarity": lambda item: (
            item["knn_score"],
            item["matched_count"],
            -item["missing_count"],
        ),
        "Hybrid explainable score": lambda item: (
            item["final_score"],
            item["knn_score"],
            item["matched_count"],
            -item["missing_count"],
        ),
    }
    if include_ml_ranker:
        rankers["Supervised ML ranker"] = lambda item: (
            item.get("ml_score", 0),
            item["final_score"],
            item["knn_score"],
            -item["missing_count"],
        )
    return rankers


def interpret_generalization(train_f1: float, test_f1: float) -> str:
    gap = train_f1 - test_f1
    if test_f1 < 0.8:
        return "Likely underfitting or needs tuning"
    if gap > 0.08:
        return "Possible overfitting"
    if gap < -0.03:
        return "Test performance exceeds train split"
    return "Stable generalization"


def build_generalization_rows(model_comparison: List[Dict]) -> List[Dict]:
    rows = []
    for model in model_comparison:
        train_f1 = model["train_metrics"]["f1"]
        test_f1 = model["test_metrics"]["f1"]
        gap = round(train_f1 - test_f1, 4)
        rows.append(
            {
                "model_type": model["model_type"],
                "train_f1": train_f1,
                "test_f1": test_f1,
                "gap": gap,
                "interpretation": interpret_generalization(train_f1, test_f1),
            }
        )
    return rows


def run_evaluation(
    recommender: SmartKitchenRecommender = None,
    profile: Dict = None,
) -> Dict:
    recommender = recommender or SmartKitchenRecommender(DATA_DIR / "recipes_clean.json")
    profile = profile if profile is not None else UserPreferenceStore(DATA_DIR / "user_profile.json").snapshot()
    rankers = get_rankers(include_ml_ranker=recommender.ml_ranker is not None)
    per_ranker_results = {name: [] for name in rankers}
    query_results = []

    for query in EVALUATION_QUERIES:
        candidates = build_candidates(recommender, profile, query)
        query_rows = []

        for name, ranker in rankers.items():
            ranked = rank_candidates(candidates, ranker)
            result = evaluate_ranker(name, ranked)
            per_ranker_results[name].append(result)
            query_rows.append(result)

        query_results.append(
            {
                "label": ", ".join(query),
                "ingredients": query,
                "candidate_count": len(candidates),
                "rows": query_rows,
            }
        )

    model_comparison = recommender.ml_ranker.get("model_comparison", []) if recommender.ml_ranker else []

    return {
        "top_k": TOP_K,
        "query_count": len(EVALUATION_QUERIES),
        "ml_ranker_available": recommender.ml_ranker is not None,
        "ml_model_type": recommender.ml_ranker.get("default_model_type") if recommender.ml_ranker else None,
        "ml_training": recommender.ml_ranker.get("training") if recommender.ml_ranker else None,
        "ml_feature_weights": recommender.ml_ranker.get("feature_weights", []) if recommender.ml_ranker else [],
        "ml_model_comparison": model_comparison,
        "ml_generalization_rows": build_generalization_rows(model_comparison),
        "relevance_proxy": "matched>=2, missing<=6, coverage>=0.25, knn>=0.10",
        "candidate_filter": f"knn>={MIN_SCORE}, matched>={MIN_MATCHED_COUNT}, missing<={MAX_MISSING_COUNT}",
        "aggregate_rows": [aggregate(results) for results in per_ranker_results.values()],
        "query_results": query_results,
    }


def main() -> None:
    report = run_evaluation()

    print(f"Offline recommender evaluation on {report['query_count']} pantry queries")
    print(f"Relevance proxy: {report['relevance_proxy']}")
    print(f"Candidate filter: {report['candidate_filter']}")
    print()

    for query in report["query_results"]:
        print(f"Query: {query['label']} ({query['candidate_count']} candidates)")
        print_table(query["rows"])
        print()

    print("Aggregate results")
    print_table(report["aggregate_rows"])


if __name__ == "__main__":
    main()
