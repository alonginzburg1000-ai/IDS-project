import os

import numpy as np
import pandas as pd

from ml_from_scratch.layers import ActivationReLU, ActivationSigmoid, LayerDense
from ml_from_scratch.losses import BinaryCrossEntropy
from ml_from_scratch.model import DFFNN


def load_config(config_path="ml_from_scratch/config.json"):
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Ч”Ч§Ч•Ч‘ЧҐ ЧўЧ•Ч©Ч” 4 Ч“Ч‘ЧЁЧ™Чќ:
# ЧћЧ›Ч™Чџ ЧђЧЄ Ч”Ч“ЧђЧЧ”
# Ч‘Ч•Ч Ч” ЧћЧ•Ч“Чњ
# ЧћЧ’Ч“Ч™ЧЁ LOSS
# ЧњЧ•ЧњЧђЧЄ ЧђЧ™ЧћЧ•Чџ


def accuracy_binary(y_pred, y_true, threshold=0.5):
    # Ч”ЧћЧЁЧЄ Ч”ЧЎЧЄЧ‘ЧЁЧ•Ч™Ч•ЧЄ ЧњЧђЧ•ЧЄЧ•ЧЄ 0/1 ЧњЧ¤Ч™ ЧЎЧЈ
    y_pred_class = (y_pred >= threshold).astype(int)
    # Ч›Ч“Ч™ Ч©Ч©Ч Ч™ Ч”Ч•Ч§ЧЧ•ЧЁЧ™Чќ Ч™Ч”Ч™Ч• Ч‘Ч“Ч™Ч•Ч§ Ч‘ЧђЧ•ЧЄЧ” Ч¦Ч•ЧЁЧ”
    y_true = y_true.reshape(y_pred_class.shape)
    return float(np.mean(y_pred_class == y_true))


def train_val_split_df(df, val_ratio=0.2, seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")
    # Ч™Ч¦Ч™ЧЁЧЄ ЧћЧ—Ч•ЧњЧњ ЧђЧ§ЧЁЧђЧ™ Ч§Ч‘Ч•Чў
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    # Ч—ЧњЧ•Ч§Ч” 80/20 Ч›Ч‘ЧЁЧ™ЧЁЧЄ ЧћЧ—Ч“Чњ
    split = int(len(df) * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split produced empty train or validation set. Adjust val_ratio.")
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def preprocess_binary(df_train, df_val):
    # Ч”Ч Ч—Ч” Ч©Ч‘Ч¤ЧЁЧ™ЧЎЧ” Ч©ЧњЧ Ч•: ЧўЧћЧ•Ч“Ч” ЧњЧ¤Ч Ч™ ЧђЧ—ЧЁЧ•Ч Ч” = LABEL, ЧђЧ—ЧЁЧ•Ч Ч” = DIFFICULTY
    label_col = df_train.columns[-2]
    difficulty_col = df_train.columns[-1]

    # Ч Ч™Ч§Ч•Ч™ Ч•ЧњЧ™Ч™Ч‘ЧњЧ™Чќ
    y_train_raw = df_train[label_col].astype(str).str.strip().str.lower()
    y_val_raw = df_val[label_col].astype(str).str.strip().str.lower()
    y_train = (y_train_raw != "normal").astype(int).values.reshape(-1, 1)
    y_val = (y_val_raw != "normal").astype(int).values.reshape(-1, 1)

    # Ч‘Ч Ч™Ч™ЧЄ X Ч‘ЧњЧ™ ЧўЧћЧ•Ч“Ч•ЧЄ label Ч•-difficulty
    X_train_df = df_train.drop(columns=[label_col, difficulty_col])
    X_val_df = df_val.drop(columns=[label_col, difficulty_col])

    # One-Hot Encoding ЧњЧўЧћЧ•Ч“Ч•ЧЄ Ч§ЧЧ’Ч•ЧЁЧ™Ч•ЧЄ
    cat_cols = [1, 2, 3]
    X_train_df = pd.get_dummies(X_train_df, columns=cat_cols)
    X_val_df = pd.get_dummies(X_val_df, columns=cat_cols)
    # Ч™Ч™Ч©Ч•ЧЁ ЧўЧћЧ•Ч“Ч•ЧЄ Ч‘Ч™Чџ train Чњ-val Ч›Ч“Ч™ Ч©Ч™Ч¦ЧђЧ• Ч–Ч”Ч•ЧЄ Ч‘ЧћЧ™ЧћЧ“Ч™Чќ
    X_train_df, X_val_df = X_train_df.align(X_val_df, join="left", axis=1, fill_value=0)

    # Ч©ЧћЧ™ЧЁЧЄ Ч©ЧћЧ•ЧЄ Ч¤Ч™Ч¦'ЧЁЧ™Чќ
    #XX: Ensure feature names are normalized to strings for consistent reuse in inference/training artifacts
    feature_names = pd.Index(X_train_df.columns).astype(str).to_numpy()

    # Ч”ЧћЧЁЧ” ЧњЦѕNumPy float32
    X_train = X_train_df.astype(np.float32).values
    X_val = X_val_df.astype(np.float32).values

    # ЧЎЧЧ Ч“ЧЁЧЧ™Ч–Ч¦Ч™Ч” (Normalization) ЧўЧњ Ч‘ЧЎЧ™ЧЎ train Ч‘ЧњЧ‘Ч“
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, y_train, y_val, mean, std, feature_names


def preprocess_test_binary(df_test, feature_names, mean, std):
    label_col = df_test.columns[-2]
    difficulty_col = df_test.columns[-1]

    y_test_raw = df_test[label_col].astype(str).str.strip().str.lower()
    y_test = (y_test_raw != "normal").astype(int).values.reshape(-1, 1)

    X_test_df = df_test.drop(columns=[label_col, difficulty_col])
    cat_cols = [1, 2, 3]
    X_test_df = pd.get_dummies(X_test_df, columns=cat_cols)
    #XX: Convert inference columns to strings and fail if required features are missing
    expected_features = pd.Index(feature_names).astype(str)
    X_test_df.columns = X_test_df.columns.astype(str)
    missing_features = [feat for feat in expected_features if feat not in X_test_df.columns]
    if len(missing_features) == len(expected_features):
        raise ValueError(
            "Feature mismatch at test preprocessing. No common feature names between train features and test prepared columns."
        )
    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} train features not present in test input columns; they will be filled with 0.")
    X_test_df = X_test_df.reindex(columns=expected_features, fill_value=0)

    X_test = X_test_df.astype(np.float32).values
    X_test = (X_test - mean) / std

    return X_test, y_test


def evaluate_split(X, y_true, model, loss_fn):
    y_pred = model.forward(X)
    loss = loss_fn.forward(y_pred, y_true)
    acc = accuracy_binary(y_pred, y_true)
    return loss, acc


def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true_vec = y_true.reshape(-1).astype(int)
    y_prob_vec = y_prob.reshape(-1)
    y_pred_vec = (y_prob_vec >= threshold).astype(int)

    tp = int(np.sum((y_true_vec == 1) & (y_pred_vec == 1)))
    tn = int(np.sum((y_true_vec == 0) & (y_pred_vec == 0)))
    fp = int(np.sum((y_true_vec == 0) & (y_pred_vec == 1)))
    fn = int(np.sum((y_true_vec == 1) & (y_pred_vec == 0)))

    precision_den = tp + fp
    recall_den = tp + fn

    precision = float(tp / precision_den) if precision_den > 0 else 0.0
    recall = float(tp / recall_den) if recall_den > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "acc": float(np.mean(y_pred_vec == y_true_vec)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": int(y_true_vec.shape[0]),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_test(df_test, model, loss_fn, feature_names, mean, std, threshold=0.5):
    X_test, y_test = preprocess_test_binary(df_test, feature_names, mean, std)
    y_pred = model.forward(X_test)
    loss = loss_fn.forward(y_pred, y_test)
    metrics = binary_classification_metrics(y_test, y_pred, threshold=threshold)
    metrics["loss"] = float(loss)
    return metrics


def save_learning_curves(train_losses, train_accs, val_losses, val_accs, artifacts_dir):
    import csv

    epochs = np.arange(1, len(train_losses) + 1)
    curve_csv_path = os.path.join(artifacts_dir, "learning_curves.csv")
    with open(curve_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(epochs)):
            writer.writerow([int(epochs[i]), float(train_losses[i]), float(train_accs[i]), float(val_losses[i]), float(val_accs[i])])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, train_losses, label="train_loss")
        axes[0].plot(epochs, val_losses, label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, train_accs, label="train_acc")
        axes[1].plot(epochs, val_accs, label="val_acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        fig.tight_layout()
        curve_path = os.path.join(artifacts_dir, "learning_curves.png")
        fig.savefig(curve_path)
        plt.close(fig)
        print(f"Saved learning curves image: {curve_path}")
    except Exception:
        print(f"matplotlib not available. Saved numeric curve data: {curve_csv_path}")
        chart_lines = 12
        print("ASCII learning curves (loss):")
        max_loss = max(train_losses + val_losses)
        min_loss = min(train_losses + val_losses)
        loss_range = max(max_loss - min_loss, 1e-12)
        loss_scale = (chart_lines - 1) / loss_range

        for row in range(chart_lines):
            level = max_loss - row / max(loss_scale, 1e-12)
            chars = []
            for tr, vr in zip(train_losses, val_losses):
                ch = " "
                t_row = int((max_loss - tr) * loss_scale)
                v_row = int((max_loss - vr) * loss_scale)
                if row == t_row and row == v_row:
                    ch = "X"
                elif row == t_row:
                    ch = "T"
                elif row == v_row:
                    ch = "V"
                chars.append(ch)
            print(f"{level:6.3f} | " + "".join(chars))

        print("Legend: T=train_loss  V=val_loss")


def save_eval_report(json_path, csv_path, config, val_metrics, test_metrics, model_path, artifacts_dir):
    import csv
    import json

    model_info = {"best_or_final_model_path": model_path, "artifact_dir": artifacts_dir}
    report = {
        "config": config,
        "model": model_info,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "loss", "acc", "precision", "recall", "f1", "rows", "tn", "fp", "fn", "tp"])
        writer.writerow([
            "val",
            report["val"]["loss"],
            report["val"]["acc"],
            report["val"]["precision"],
            report["val"]["recall"],
            report["val"]["f1"],
            report["val"]["rows"],
            report["val"]["confusion"]["tn"],
            report["val"]["confusion"]["fp"],
            report["val"]["confusion"]["fn"],
            report["val"]["confusion"]["tp"],
        ])
        if test_metrics is None:
            writer.writerow(["test", "", "", "", "", "", "", "", "", "", ""])
        else:
            writer.writerow([
                "test",
                test_metrics["loss"],
                test_metrics["acc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
                test_metrics["rows"],
                test_metrics["confusion"]["tn"],
                test_metrics["confusion"]["fp"],
                test_metrics["confusion"]["fn"],
                test_metrics["confusion"]["tp"],
            ])


def main():
    config = load_config()

    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})

    seed = training_cfg.get("seed", 42)
    val_ratio = training_cfg.get("val_ratio", 0.2)
    lr = training_cfg.get("lr", 0.01)
    threshold = training_cfg.get("threshold", 0.5)
    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 64))
    hidden1 = int(model_cfg.get("hidden1", 64))
    hidden2 = int(model_cfg.get("hidden2", 32))

    train_path = paths_cfg.get("train_path", "data/raw/KDDTrain+.txt")
    test_path = paths_cfg.get("test_path", "data/raw/KDDTest+.txt")
    artifacts_dir = paths_cfg.get("artifacts_dir", "artifacts")

    # Seed ЧњЧЁЧ Ч“Ч•ЧћЧњЧ™Ч•ЧЄ
    np.random.seed(seed)

    os.makedirs(artifacts_dir, exist_ok=True)

    # Ч§ЧЁЧ™ЧђЧЄ Ч”Ч“ЧђЧЧ”
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
    df = pd.read_csv(train_path, header=None)
    print("Raw df shape:", df.shape)

    # Split + Preprocess
    # y = 0 ЧђЧќ normal, ЧђЧ—ЧЁЧЄ 1
    # ЧўЧ•Ч©Ч” one-hot ЧњЧўЧћЧ•Ч“Ч•ЧЄ Ч§ЧЧ’Ч•ЧЁЧ™Ч•ЧЄ
    # ЧћЧ™Ч™Ч©ЧЁ ЧўЧћЧ•Ч“Ч•ЧЄ train/val
    df_train, df_val = train_val_split_df(df, val_ratio=val_ratio, seed=seed)
    X_train, X_val, y_train, y_val, mean, std, feature_names = preprocess_binary(df_train, df_val)

    print("Shapes after preprocessing:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)

    # Ч‘Ч Ч™Ч™ЧЄ Ч”ЧЁЧ©ЧЄ
    model = DFFNN([
        LayerDense(n_inputs=X_train.shape[1], n_neurons=hidden1),
        ActivationReLU(),
        LayerDense(n_inputs=hidden1, n_neurons=hidden2),
        ActivationReLU(),
        LayerDense(n_inputs=hidden2, n_neurons=1),
        ActivationSigmoid(),
    ])

    # Loss Ч©Чњ BCE
    loss_fn = BinaryCrossEntropy()

    #XX: Keep best-model bookkeeping so report matches the saved best checkpoint
    best_val_loss = float("inf")
    best_epoch = None
    best_val_metrics = None
    best_model_path = os.path.join(artifacts_dir, "binary_model_weights_best.npz")

    # ЧђЧ™ЧћЧ•Чџ ЧўЧќ mini-batches Ч‘Ч›Чњ epoch
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_losses = []
        epoch_accs = []
        for start in range(0, len(X_shuf), batch_size):
            end = start + batch_size
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            y_pred = model.forward(Xb)
            loss = loss_fn.forward(y_pred, yb)

            dL_dPred = loss_fn.backward(y_pred, yb)
            model.backward(dL_dPred)
            model.update(lr)

            epoch_losses.append(loss)
            epoch_accs.append(accuracy_binary(y_pred, yb))

        # Ч•ЧњЧ™Ч“Ч¦Ч™Ч” Ч‘ЧЎЧ•ЧЈ epoch
        y_val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(y_val_pred, y_val)
        val_metrics = binary_classification_metrics(y_val, y_val_pred, threshold=threshold)
        val_acc = val_metrics["acc"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(best_model_path)
            best_epoch = epoch
            best_val_metrics = {
                "epoch": epoch,
                "loss": float(val_loss),
                **val_metrics,
            }

        train_loss_history.append(np.mean(epoch_losses))
        train_acc_history.append(np.mean(epoch_accs))
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(epoch_losses):.4f} train_acc={np.mean(epoch_accs):.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    save_learning_curves(train_loss_history, train_acc_history, val_loss_history, val_acc_history, artifacts_dir)

    # Ч©ЧћЧ™ЧЁЧ” ЧњЧ§Ч‘Ч¦Ч™Чќ
    model_path = os.path.join(artifacts_dir, "binary_model_weights.npz")
    preprocess_path = os.path.join(artifacts_dir, "binary_preprocess.npz")
    model.save_weights(model_path)
    np.savez(preprocess_path, mean=mean, std=std, feature_names=feature_names)

    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
        model_for_report = best_model_path
    else:
        model.load_weights(model_path)
        model_for_report = model_path
        best_val_metrics = {
            "epoch": epochs,
            "loss": float(val_loss_history[-1]),
            "acc": float(val_acc_history[-1]),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rows": int(X_val.shape[0]),
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }

    #XX: If no checkpoint improvement happened, still report last-epoch metrics
    if best_val_metrics is None:
        best_val_pred = model.forward(X_val)
        fallback_metrics = binary_classification_metrics(y_val, best_val_pred, threshold=threshold)
        best_val_metrics = {
            "epoch": epochs,
            "loss": float(val_loss_history[-1]),
            **fallback_metrics,
        }

    test_metrics = None
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, header=None)
        test_metrics = evaluate_test(df_test, model, loss_fn, feature_names, mean, std, threshold=threshold)
        #XX: print extra quality metrics for binary IDS reporting
        print(
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
        )
    else:
        print(f"Test file not found: {test_path}")

    save_eval_report(
        os.path.join(artifacts_dir, "train_eval_report.json"),
        os.path.join(artifacts_dir, "train_eval_report.csv"),
        config,
        best_val_metrics,
        test_metrics,
        model_for_report,
        artifacts_dir,
    )


if __name__ == "__main__":
    main()
