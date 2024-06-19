from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, \
    mean_absolute_error, accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef
import numpy as np


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    # print(labels.shape, logits.shape)
    # logits = logits[0]
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)