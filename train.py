import argparse

from src.model import CLASSIFIERS, train_all_models, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fake news detection model.")
    parser.add_argument("--data", default="sample_news.csv", help="CSV file with text and label columns.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "logistic_regression", "naive_bayes", "svm", "perceptron"],
        help="Classifier to train. Use all to compare every classifier.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split size between 0.15 and 0.40.")
    args = parser.parse_args()

    if args.model == "all":
        results, best_model = train_all_models(args.data, test_size=args.test_size)
        print("Classifier Results")
        for result in sorted(results, key=lambda item: (item.f1, item.accuracy), reverse=True):
            best_marker = " (best)" if result.model_name == best_model else ""
            cv_text = "N/A" if result.cv_accuracy_mean is None else f"{result.cv_accuracy_mean:.3f}"
            auc_text = "N/A" if result.roc_auc is None else f"{result.roc_auc:.3f}"
            print(
                f"- {result.display_name}: "
                f"accuracy={result.accuracy:.3f}, "
                f"precision={result.precision:.3f}, "
                f"recall={result.recall:.3f}, "
                f"f1={result.f1:.3f}, "
                f"roc_auc={auc_text}, "
                f"cv_accuracy={cv_text}{best_marker}"
            )
        print(f"\nBest model saved for prediction: {CLASSIFIERS[best_model]}")
        print("Ensemble model saved for optional prediction.\n")
        for result in sorted(results, key=lambda item: (item.f1, item.accuracy), reverse=True):
            print(f"--- {result.display_name} ---")
            if result.best_params:
                print("Best params:", result.best_params)
            print(result.report)
    else:
        result = train_model(args.data, args.model, test_size=args.test_size)
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Precision: {result.precision:.3f}")
        print(f"Recall: {result.recall:.3f}")
        print(f"F1: {result.f1:.3f}")
        if result.roc_auc is not None:
            print(f"ROC-AUC: {result.roc_auc:.3f}")
        print("Labels:", ", ".join(result.labels))
        print(result.report)


if __name__ == "__main__":
    main()
