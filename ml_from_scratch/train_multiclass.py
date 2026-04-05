import csv
import json
import os

import numpy as np
import pandas as pd

from ml_from_scratch.layers import ActivationReLU, LayerDense
from ml_from_scratch.losses import SoftmaxCrossEntropy
from ml_from_scratch.model import DFFNN
from ml_from_scratch.multiclass_utils import (
    FAMILY_NAMES,
    extract_attack_rows_and_targets,
    load_json_config,
    load_saved_preprocess,
    multiclass_accuracy,
    multiclass_metrics,
    prepare_features,
    save_label_map,
    stratified_split_df,
)
"""
        הקובץ הזה:

טוען קונפיג
טוען את ה־preprocessing של המודל הבינארי
קורא את KDDTrain+
מסנן רק דוגמאות של attack
מחלק ל־train / validation
מכין פיצ'רים
בונה מודל רב־מחלקתי
מאמן אותו על batches
שומר best model + final model
מעריך על validation ועל test
שומר דוחות וגרפים
        """
#בניית המודל
def build_multiclass_model(n_inputs, hidden1=100, hidden2=100, num_classes=4):
    return DFFNN([
        LayerDense(n_inputs=n_inputs, n_neurons=hidden1),
        ActivationReLU(),
        LayerDense(n_inputs=hidden1, n_neurons=hidden2),
        ActivationReLU(),
        LayerDense(n_inputs=hidden2, n_neurons=num_classes),
    ])

#פונקציה אחת מסודרת להערכת המודל
"""
forward pass
חישוב loss
חישוב metrics
החזרת הכל כ־dict
        """
def evaluate_multiclass(X, y_true, model, loss_fn):
    logits = model.forward(X)
    loss = loss_fn.forward(logits, y_true)
    metrics = multiclass_metrics(y_true, logits, label_names=FAMILY_NAMES)
    metrics["loss"] = float(loss)
    return metrics

#הפונקציה הזו שומרת את ההיסטוריה של האימון
def save_learning_curves(train_losses, train_accs, val_losses, val_accs, artifacts_dir):
    curve_csv_path = os.path.join(artifacts_dir, "multiclass_learning_curves.csv")
    with open(curve_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch_idx in range(len(train_losses)):
            #גם אם אין גרף, עדיין יש נתונים מספריים שאפשר לנתח
            writer.writerow([
                epoch_idx + 1,
                float(train_losses[epoch_idx]),
                float(train_accs[epoch_idx]),
                float(val_losses[epoch_idx]),
                float(val_accs[epoch_idx]),
            ])

#כדי שהקוד לא יקרוס אם matplotlib לא מותקן.
    try:
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(train_losses) + 1)
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
        curve_path = os.path.join(artifacts_dir, "multiclass_learning_curves.png")
        fig.savefig(curve_path)
        plt.close(fig)
        print(f"Saved learning curves image: {curve_path}")
    except Exception:
        print(f"matplotlib not available. Saved numeric curve data: {curve_csv_path}")

#פונקציה הזו שומרת את תוצאות ההערכה בכמה פורמטים
def save_eval_report(json_path, csv_path, per_class_csv_path, config, val_metrics, test_metrics, model_path, artifacts_dir):
    report = {
        "config": config,
        "model": {"best_or_final_model_path": model_path, "artifact_dir": artifacts_dir},
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        #CSV כללי
        writer = csv.writer(f)
        writer.writerow(["split", "loss", "acc", "macro_precision", "macro_recall", "macro_f1", "rows"])
        for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
            if metrics is None:
                writer.writerow([split_name, "", "", "", "", "", ""])
                continue
            #CSV לפי מחלקה
            writer.writerow([
                split_name,
                metrics["loss"],
                metrics["acc"],
                metrics["macro_precision"],
                metrics["macro_recall"],
                metrics["macro_f1"],
                metrics["rows"],
            ])

    with open(per_class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "precision", "recall", "f1", "support"])
        for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
            if metrics is None:
                continue
            for label_name in metrics["label_order"]:
                class_metrics = metrics["per_class"][label_name]
                writer.writerow([
                    split_name,
                    label_name,
                    class_metrics["precision"],
                    class_metrics["recall"],
                    class_metrics["f1"],
                    class_metrics["support"],
                ])

#טעינת config
def main():
    config = load_json_config()
    paths_cfg = config.get("paths", {})
    multiclass_cfg = config.get("multiclass", {})
#קריאת paths
    train_path = paths_cfg.get("train_path", "data/raw/KDDTrain+.txt")
    test_path = paths_cfg.get("test_path", "data/raw/KDDTest+.txt")
    artifacts_dir = paths_cfg.get("artifacts_dir", "artifacts")
#קריאת hyperparameters
#כל הפרמטרים החשובים באים מהקונפיג
    seed = int(multiclass_cfg.get("seed", 42))
    val_ratio = float(multiclass_cfg.get("val_ratio", 0.2))
    epochs = int(multiclass_cfg.get("epochs", 20))
    batch_size = int(multiclass_cfg.get("batch_size", 64))
    lr = float(multiclass_cfg.get("lr", 0.01))
    hidden1 = int(multiclass_cfg.get("hidden1", 100))
    hidden2 = int(multiclass_cfg.get("hidden2", 100))
    num_classes = int(multiclass_cfg.get("num_classes", len(FAMILY_NAMES)))
    #בדיקת sanity
    if num_classes != len(FAMILY_NAMES):
        raise ValueError(f"num_classes must be {len(FAMILY_NAMES)} for the configured attack families.")

#נתיבי ארטיפקטים
    preprocess_source = multiclass_cfg.get("preprocess_source", os.path.join(artifacts_dir, "binary_preprocess.npz"))
    model_path = os.path.join(artifacts_dir, "multiclass_model_weights.npz")
    best_model_path = os.path.join(artifacts_dir, "multiclass_model_weights_best.npz")
    preprocess_artifact_path = os.path.join(artifacts_dir, "multiclass_preprocess.npz")
    label_map_path = os.path.join(artifacts_dir, "multiclass_label_map.json")
    eval_json_path = os.path.join(artifacts_dir, "multiclass_eval_report.json")
    eval_csv_path = os.path.join(artifacts_dir, "multiclass_eval_report.csv")
    per_class_csv_path = os.path.join(artifacts_dir, "multiclass_per_class_report.csv")

#seed ותיקייה
    np.random.seed(seed)
    os.makedirs(artifacts_dir, exist_ok=True)
#בדיקת קובץ train
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
#טעינת preprocessing של המודל הראשון
#משתמשים באותם רכיבים של המודל הבנארי
    mean, std, feature_names = load_saved_preprocess(preprocess_source)
    df = pd.read_csv(train_path, header=None)
    #קריאת הדאטה והוצאת attack בלבד
    df_attack, y_attack = extract_attack_rows_and_targets(df)
    #חלוקה ל־train / validation
    df_train, df_val, y_train, y_val = stratified_split_df(df_attack, y_attack, val_ratio=val_ratio, seed=seed)
#preprocessing לפיצ'רים
    X_train = prepare_features(df_train, feature_names, mean, std)
    X_val = prepare_features(df_val, feature_names, mean, std)
#הדפסת shapes
    print("Multiclass shapes after preprocessing:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
#בניית המודל וה־loss
    model = build_multiclass_model(X_train.shape[1], hidden1=hidden1, hidden2=hidden2, num_classes=num_classes)
    loss_fn = SoftmaxCrossEntropy()
#משתני מעקב
    best_val_loss = float("inf")
    best_epoch = None
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
#לולאת האימון
    for epoch in range(1, epochs + 1):
        #ערבוב הדאטה בכל epoch
        idx = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]
#משתני epoch
        epoch_losses = []
        epoch_accs = []
        #מעבר על batches
        for start in range(0, len(X_shuf), batch_size):
            end = start + batch_size
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]

        #forward
            logits = model.forward(Xb)
            loss = loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward(logits, yb)
        #backward
            model.backward(dlogits)
        #update
            model.update(lr)
        #מעקב אחרי batch metrics
            epoch_losses.append(loss)
            epoch_accs.append(multiclass_accuracy(logits, yb))
    #הערכת validation בסוף כל epoch
        val_logits = model.forward(X_val)
        val_loss = loss_fn.forward(val_logits, y_val)
        val_metrics = multiclass_metrics(y_val, val_logits, label_names=FAMILY_NAMES)
        val_acc = val_metrics["acc"]
    #שמירת best model
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            model.save_weights(best_model_path)
    #שמירת history
        train_loss_history.append(float(np.mean(epoch_losses)))
        train_acc_history.append(float(np.mean(epoch_accs)))
        val_loss_history.append(float(val_loss))
        val_acc_history.append(float(val_acc))
    #הדפסת התקדמות
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(epoch_losses):.4f} train_acc={np.mean(epoch_accs):.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
#שמירת עקומת למידה
    save_learning_curves(train_loss_history, train_acc_history, val_loss_history, val_acc_history, artifacts_dir)
#שמירת המודל וה־preprocess
    model.save_weights(model_path)
    np.savez(
        preprocess_artifact_path,
        mean=mean,
        std=std,
        feature_names=feature_names,
        label_order=np.array(FAMILY_NAMES),
    )
    #שומר את מיפוי המחלקות
    save_label_map(label_map_path)

#טעינת best model לדוח
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
        model_for_report = best_model_path
    else:
        model.load_weights(model_path)
        model_for_report = model_path
#הערכה על validation
    val_metrics = evaluate_multiclass(X_val, y_val, model, loss_fn)
    val_metrics["epoch"] = best_epoch if best_epoch is not None else epochs
#הערכה על test
    test_metrics = None
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, header=None)
        df_test_attack, y_test = extract_attack_rows_and_targets(df_test)
        X_test = prepare_features(df_test_attack, feature_names, mean, std)
        test_metrics = evaluate_multiclass(X_test, y_test, model, loss_fn)
        #הדפסת תוצאות test
        print(
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )
    else:
        print(f"Test file not found: {test_path}")
#שמירת דוחות סופיים
    save_eval_report(
        eval_json_path,
        eval_csv_path,
        per_class_csv_path,
        config,
        val_metrics,
        test_metrics,
        model_for_report,
        artifacts_dir,
    )


if __name__ == "__main__":
    main()
