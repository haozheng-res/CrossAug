import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def _ensure_2d_last_dim(arr):
    """
    Flatten all dimensions except the last for any array with ndim >= 2.
    Return shape: (N_samples, n_outputs).
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError("Input array must have ndim >= 2 (last dim = features).")
    new_shape = (-1, arr.shape[-1])  # Merge all preceding dimensions into the sample dimension.

    return arr.reshape(new_shape)

def calculate_mase(y_pred, y_true, y_naive=None):
    """
    MASE (Mean Absolute Scaled Error).
    y_true: (b, l, n)
    y_pred: (b, l, n)
    y_naive: (b, l, n)
    """

    assert y_pred.ndim >= 3, 'Error in y_pred.shape...'

    if y_naive is None:
        y_naive = y_true

    if y_true.shape != y_pred.shape or y_true.shape != y_naive.shape:
        raise ValueError("The input shape must be the same.")

    # reshape to [batch, pred_len, channel]
    y_true_flat = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
    y_naive_flat = y_naive.reshape(-1, y_naive.shape[-2], y_naive.shape[-1])

    # MAE: [batch, channel]
    mae_model = np.mean(np.abs(y_true_flat - y_pred_flat), axis=1)

    # naive MAE [batch, channel]
    mae_naive = np.mean(np.abs(y_true_flat[:, 1:, :] - y_naive_flat[:, :-1, :]), axis=1)

    # avoid potential error
    mae_naive[mae_naive < 1e-5] = np.nan

    # MASE
    mase = np.nanmean(mae_model / mae_naive)

    return mase

def compute_multioutput_metrics(pred, true):
    y_true = _ensure_2d_last_dim(true)  # Convert to 2D: (N_samples, n_outputs).
    y_pred = _ensure_2d_last_dim(pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch after flattening: {y_true.shape} vs {y_pred.shape}")

    mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')  # MAE
    mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')  # MSE
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput='uniform_average') * 100  # MAPE, converted to a percentage.
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2  # SMAPE
    smape = np.mean(np.abs(y_true - y_pred) / np.where(denominator == 0, 1e-8, denominator)) * 100
    nmae = mae / np.mean(y_true)

    mase = calculate_mase(pred, true)
    mspe = 0

    return mae, mse, rmse, r2, mape, smape, nmae, mase, mspe

def metric(pred, true):
    mae, mse, rmse, r2, mape, smape, nmae, mase, mspe = compute_multioutput_metrics(pred, true)
    return mae, mse, rmse, r2, mape, smape, nmae, mase, mspe


