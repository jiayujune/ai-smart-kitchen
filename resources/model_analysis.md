# Model Analysis Notes

## Models Compared

- Logistic Regression: traditional linear classifier with interpretable learned weights.
- Decision Tree: traditional non-linear classifier with feature-threshold splits.
- MLP Neural Network: neural-network model with one hidden layer.

## Current Result Summary

The Decision Tree currently has the strongest held-out classification F1 score, so the web app serves it as the default supervised ML ranker. The hybrid explainable ranker remains strong for practical cooking utility because it explicitly penalizes missing ingredients and rewards low-shopping options.

## Interpretation

The supervised models learn from pseudo-labels generated from ingredient relevance rules. This gives the project a supervised learning pipeline, but the labels still reflect engineered assumptions rather than real user ratings. Classification metrics and recommendation-ranking metrics can disagree: a model can classify relevance well while still ranking recipes with more missing ingredients than a hand-tuned hybrid score.

## Underfitting and Overfitting Analysis

The evaluation dashboard compares train and test F1 for each supervised model to detect generalization issues.

| Model | Train F1 | Test F1 | Interpretation |
| --- | ---: | ---: | --- |
| Logistic Regression | 0.9649 | 0.9526 | Small train-test gap, stable generalization. |
| Decision Tree | close to test F1 | 0.9783 | Best held-out F1; no severe overfitting observed in the current split. |
| MLP Neural Network | lower than the tree model | 0.7874 | Likely underfitting or insufficient tuning because recall is much lower than the other models. |

The MLP has perfect precision but lower recall, meaning it is conservative: when it predicts relevance it is usually correct, but it misses many relevant recipes. This suggests the neural network needs more tuning, more training data, or a different architecture. The Decision Tree performs best on the current pseudo-labeled test split, while Logistic Regression remains valuable because its weights are easier to interpret.

## Future Improvements

- Replace pseudo-labels with real user feedback labels from clicks, likes, saves, and ignored recommendations.
- Add cross-validation and systematic hyperparameter tuning.
- Compare ranking-specific metrics such as NDCG@K or MAP@K.
- Collect more diverse evaluation pantry queries.
