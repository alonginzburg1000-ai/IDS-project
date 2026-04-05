import json
import os

import numpy as np
import pandas as pd

from ml_from_scratch.layers import ActivationReLU, ActivationSigmoid, LayerDense
from ml_from_scratch.model import DFFNN
from ml_from_scratch.multiclass_utils import (
    extract_attack_rows_and_targets,
    get_raw_labels,
    load_json_config,
    load_saved_preprocess,
    multiclass_metrics,
    prepare_features,
    softmax,
)

#בונה את הארכיטקטורה של המודל הראשון:
def build_binary_model(n_inputs, hidden1=64, hidden2=32):
    return DFFNN([
        LayerDense(n_inputs=n_inputs, n_neurons=hidden1),
        ActivationReLU(),
        LayerDense(n_inputs=hidden1, n_neurons=hidden2),
        ActivationReLU(),
        LayerDense(n_inputs=hidden2, n_neurons=1),
        ActivationSigmoid(),
    ])

#זו אותה ארכיטקטורה שכבר ראינו בקובץ האימון של המודל השני
def build_multiclass_model(n_inputs, hidden1=100, hidden2=100, num_classes=4):
    return DFFNN([
        LayerDense(n_inputs=n_inputs, n_neurons=hidden1),
        ActivationReLU(),
        LayerDense(n_inputs=hidden1, n_neurons=hidden2),
        ActivationReLU(),
        LayerDense(n_inputs=hidden2, n_neurons=num_classes),
    ])

#מחפש איזה קובץ weights להשתמש בו
def choose_weights_path(best_path, fallback_path):
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing model weights. Checked: {best_path}, {fallback_path}")


def main():
    import argparse
#שלב א — parsing של ארגומנטים
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/KDDTest+.txt")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
#בדיקה שהדאטה קיים
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Missing inference data file: {args.data}")
#טעינת config
    config = load_json_config()
    binary_model_cfg = config.get("model", {})
    multiclass_cfg = config.get("multiclass", {})
#טעינת preprocessing של המודל הבינארי
    binary_preprocess_path = os.path.join(args.artifacts, "binary_preprocess.npz")
    #בניית המודל הבינארי וטעינת המשקולות
    binary_mean, binary_std, binary_feature_names = load_saved_preprocess(binary_preprocess_path)
    binary_model = build_binary_model(
        len(binary_feature_names),
        hidden1=int(binary_model_cfg.get("hidden1", 64)),
        hidden2=int(binary_model_cfg.get("hidden2", 32)),
    )
    #טוענים את המודל הבינארי המאומן
    binary_weights_path = choose_weights_path(
        os.path.join(args.artifacts, "binary_model_weights_best.npz"),
        os.path.join(args.artifacts, "binary_model_weights.npz"),
    )
    binary_model.load_weights(binary_weights_path)
#קריאת הדאטה והכנתו למודל 1
    df = pd.read_csv(args.data, header=None)
    X_binary = prepare_features(df, binary_feature_names, binary_mean, binary_std)
#חיזוי בינארי
    binary_probs = binary_model.forward(X_binary).reshape(-1)
    binary_pred = (binary_probs >= args.threshold).astype(int)
#קריאת labels אמיתיים אם קיימים
    labels = get_raw_labels(df)
#הדפסת סיכום הבינארי
    print(f"rows={len(df)}")
    print(f"binary_model={binary_weights_path}")
    print(f"binary_pred_normal={int((binary_pred == 0).sum())}")
    print(f"binary_pred_attack={int((binary_pred == 1).sum())}")
#אם יש labels
    if labels is not None:
        binary_true = (labels != "normal").astype(int).to_numpy()
        binary_acc = float(np.mean(binary_pred == binary_true))
        print(f"binary_accuracy={binary_acc:.4f}")
#בדיקה האם המודל הרב־מחלקתי זמין
    multiclass_preprocess_path = os.path.join(args.artifacts, "multiclass_preprocess.npz")
    multiclass_weights_path = os.path.join(args.artifacts, "multiclass_model_weights.npz")
    multiclass_best_weights_path = os.path.join(args.artifacts, "multiclass_model_weights_best.npz")
    multiclass_label_map_path = os.path.join(args.artifacts, "multiclass_label_map.json")
#בודק אם כל מה שצריך לשלב השני קיים
    multiclass_ready = (
        os.path.exists(multiclass_preprocess_path)
        and (os.path.exists(multiclass_best_weights_path) or os.path.exists(multiclass_weights_path))
        and os.path.exists(multiclass_label_map_path)
    )
#הקוד עדיין יכול להריץ רק את המודל הבינארי ולא לקרוס
    if not multiclass_ready:
        print("multiclass_artifacts_found=False")
        return
#טעינת המודל הרב־מחלקתי
    multiclass_mean, multiclass_std, multiclass_feature_names = load_saved_preprocess(multiclass_preprocess_path)
    with open(multiclass_label_map_path, "r", encoding="utf-8") as f:
    #כדי לדעת איך להמיר אינדקסים לשמות
        label_map = json.load(f)
    family_names = label_map["family_names"]
#טוען את המודל השני
    multiclass_model = build_multiclass_model(
        len(multiclass_feature_names),
        hidden1=int(multiclass_cfg.get("hidden1", 100)),
        hidden2=int(multiclass_cfg.get("hidden2", 100)),
        num_classes=int(multiclass_cfg.get("num_classes", len(family_names))),
    )
    selected_multiclass_weights = choose_weights_path(multiclass_best_weights_path, multiclass_weights_path)
    multiclass_model.load_weights(selected_multiclass_weights)
#רק שורות שהמודל הראשון חזה כ־attack ממשיכות לשלב השני
    predicted_attack_positions = np.flatnonzero(binary_pred == 1)
#אם אין בכלל תקיפות חזויות
    if predicted_attack_positions.size == 0:
        print("multiclass_model_loaded=True")
        print("predicted_attack_rows=0")
        return
#הכנת הדאטה לשלב 2
    predicted_attack_df = df.iloc[predicted_attack_positions].copy()
    X_multiclass = prepare_features(predicted_attack_df, multiclass_feature_names, multiclass_mean, multiclass_std)
#חיזוי multiclass
    multiclass_logits = multiclass_model.forward(X_multiclass)
    multiclass_probs = softmax(multiclass_logits)
    multiclass_pred_idx = np.argmax(multiclass_probs, axis=1)
    multiclass_pred_names = [family_names[int(idx)] for idx in multiclass_pred_idx]
#ספירת כמות תחזיות לכל משפחה
    family_counts = {}
    for family_name in multiclass_pred_names:
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
#בניית דוגמאות פלט
    sample_records = []
    for local_idx in range(min(args.limit, predicted_attack_positions.size)):
        row_idx = int(predicted_attack_positions[local_idx])
        family_name = multiclass_pred_names[local_idx]
        family_prob = float(multiclass_probs[local_idx, multiclass_pred_idx[local_idx]])
        #לכל שורה שומרים:
        sample_records.append({
            "row_index": row_idx,
            "binary_prob": float(binary_probs[row_idx]),
            "attack_family": family_name,
            "attack_family_prob": family_prob,
        })
#הדפסת תוצאות המודל השני
    print("multiclass_model_loaded=True")
    print(f"multiclass_model={selected_multiclass_weights}")
    print(f"attack_family_count_pred={json.dumps(family_counts, ensure_ascii=False)}")
    print(f"sample_pipeline_pred={json.dumps(sample_records, ensure_ascii=False)}")
#הערכה אמיתית של המודל השני אם יש labels
    if labels is not None:
        true_attack_df, y_true_family = extract_attack_rows_and_targets(df)
        X_true_attack = prepare_features(true_attack_df, multiclass_feature_names, multiclass_mean, multiclass_std)
        true_attack_logits = multiclass_model.forward(X_true_attack)
        attack_metrics = multiclass_metrics(y_true_family, true_attack_logits, label_names=family_names)
        print(f"attack_only_acc={attack_metrics['acc']:.4f}")
        print(f"attack_only_macro_f1={attack_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
