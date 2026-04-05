import os

import numpy as np
import pandas as pd

from ml_from_scratch.layers import ActivationReLU, ActivationSigmoid, LayerDense
from ml_from_scratch.model import DFFNN


def preprocess_for_infer(df, feature_names, mean, std):
    label_col = df.columns[-2]
    difficulty_col = df.columns[-1]

    y_true_raw = df[label_col].astype(str).str.strip().str.lower()
    y_true = (y_true_raw != "normal").astype(int).values.reshape(-1, 1)

    X_df = df.drop(columns=[label_col, difficulty_col])
    cat_cols = [1, 2, 3]
    X_df = pd.get_dummies(X_df, columns=cat_cols)
    #XX: Keep deterministic feature alignment for inference and fail early on mismatches
    expected_features = pd.Index(feature_names).astype(str)
    X_df.columns = X_df.columns.astype(str)
    missing_features = [feat for feat in expected_features if feat not in X_df.columns]
    if len(missing_features) == len(expected_features):
        raise ValueError(
            "Inference feature mismatch. No common feature names between saved features and incoming test columns."
        )
    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} saved features are missing in test input and will be filled with 0.")
    X_df = X_df.reindex(columns=expected_features, fill_value=0)
    X = X_df.astype(np.float32).values
    X = (X - mean) / std
    return X, y_true


def accuracy_binary(y_pred, y_true, threshold=0.5):
    y_pred_class = (y_pred >= threshold).astype(int)
    y_true = y_true.reshape(y_pred_class.shape)
    return float(np.mean(y_pred_class == y_true))


def build_model(n_inputs):
    return DFFNN([
        LayerDense(n_inputs=n_inputs, n_neurons=64),
        ActivationReLU(),
        LayerDense(n_inputs=64, n_neurons=32),
        ActivationReLU(),
        LayerDense(n_inputs=32, n_neurons=1),
        ActivationSigmoid(),
    ])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/KDDTest+.txt")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    preprocess_path = os.path.join(args.artifacts, "binary_preprocess.npz")
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError(f"Missing preprocess file: {preprocess_path}")

    pre = np.load(preprocess_path, allow_pickle=True)
    mean = pre["mean"]
    std = pre["std"]
    feature_names = pre["feature_names"]

    model = build_model(len(feature_names))
    best_model_path = os.path.join(args.artifacts, "binary_model_weights_best.npz")
    model_path = os.path.join(args.artifacts, "binary_model_weights.npz")

    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
        print(f"Loaded model: {best_model_path}")
    elif os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Loaded model: {model_path}")
    else:
        raise FileNotFoundError("No model weights found. Run training first.")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Missing inference data file: {args.data}")

    df = pd.read_csv(args.data, header=None)
    X, y_true = preprocess_for_infer(df, feature_names, mean, std)
    y_prob = model.forward(X)
    y_pred = (y_prob >= 0.5).astype(int)

    y_prob_flat = y_prob.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)

    acc = accuracy_binary(y_prob, y_true)
    true_names = np.where(y_true_flat == 0, "normal", "attack")
    pred_names = np.where(y_pred_flat == 0, "normal", "attack")

    print(f"rows={len(df)}")
    print(f"accuracy={acc:.4f}")
    print(f"normal_count_pred={int((y_pred_flat == 0).sum())}")
    print(f"attack_count_pred={int((y_pred_flat == 1).sum())}")
    print(f"sample_true={true_names[:5].tolist()}")
    print(f"sample_pred={pred_names[:5].tolist()}")
    print(f"sample_attack_prob={y_prob_flat[:5].tolist()}")


if __name__ == "__main__":
    main()
