# Model Analysis Notes

## Models Compared

- Logistic Regression: traditional linear classifier with interpretable learned weights.
- Decision Tree: traditional non-linear classifier with feature-threshold splits.
- MLP Neural Network: neural-network model with one hidden layer.

## Current Result Summary

The Decision Tree currently has the strongest held-out classification F1 score, so the web app serves it as the default supervised ML ranker. The hybrid explainable ranker remains strong for practical cooking utility because it explicitly penalizes missing ingredients and rewards low-shopping options.

## Interpretation

The supervised models learn from pseudo-labels generated from ingredient relevance rules. This gives the project a supervised learning pipeline, but the labels still reflect engineered assumptions rather than real user ratings. Classification metrics and recommendation-ranking metrics can disagree: a model can classify relevance well while still ranking recipes with more missing ingredients than a hand-tuned hybrid score.

## Future Improvements

- Replace pseudo-labels with real user feedback labels from clicks, likes, saves, and ignored recommendations.
- Add cross-validation and systematic hyperparameter tuning.
- Compare ranking-specific metrics such as NDCG@K or MAP@K.
- Collect more diverse evaluation pantry queries.

