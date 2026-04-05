import json
import os

import numpy as np
import pandas as pd


FEATURE_COUNT = 41#יש לנו 41 פיצרים בדאטה סט 
FAMILY_NAMES = ["dos", "probe", "r2l", "u2r"]#שמות ההתקפות אותם נסווג
#מיפוי שם  משפחה לאינדקס :DOS -0 PROB-1 והלאה
FAMILY_TO_INDEX = {name: idx for idx, name in enumerate(FAMILY_NAMES)}
#מיפוי הפוך
INDEX_TO_FAMILY = {idx: name for name, idx in FAMILY_TO_INDEX.items()}
ATTACK_FAMILY_MAP = {
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "worm": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}

#פונקציה שטוענת קובץ config.json ומחזירה אותו כאובייקט Python.
def load_json_config(config_path="ml_from_scratch/config.json"):
    #בודקים אם קונפינג קיים
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

#בודקת אם הדאטה שלם 41+2 שאנחנו מורידים בפרי פרוסס
def has_target_columns(df):
    return df.shape[1] == FEATURE_COUNT + 2

#מוציאה רק את הפיצ'רים (בלי label ו־difficulty)
def get_feature_frame(df):
    if has_target_columns(df):
        label_col = df.columns[-2]
        difficulty_col = df.columns[-1]
        return df.drop(columns=[label_col, difficulty_col])
    return df.copy()

#מחזירה את labels הגולמיים
def get_raw_labels(df):
    if not has_target_columns(df):
        return None
    return df.iloc[:, -2].astype(str).str.strip().str.lower()

#טעינת preprocessing מהאימון




#טוענת את מה שלמדת בזמן האימון STD MEAN FEUTARE NAME
def load_saved_preprocess(preprocess_path):
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError(f"Missing preprocess file: {preprocess_path}")
    #טוען קובץ
    pre = np.load(preprocess_path, allow_pickle=True)
    mean = pre["mean"]
    std = pre["std"]
    feature_names = pre["feature_names"]
    return mean, std, feature_names


def prepare_features(df, feature_names, mean, std):
    #לקחת רק פיצ'רים
    X_df = get_feature_frame(df)
    #One-Hot Encoding
    cat_cols = [1, 2, 3]
    X_df = pd.get_dummies(X_df, columns=cat_cols)
    #התאמת פיצ'רים למה שהיה באימון
    expected_features = pd.Index(feature_names).astype(str)
    #reindex
    X_df.columns = X_df.columns.astype(str)
    missing_features = [feat for feat in expected_features if feat not in X_df.columns]
    if len(missing_features) == len(expected_features):
        raise ValueError(
            "Feature mismatch. No common feature names between saved features and prepared columns."
        )
    X_df = X_df.reindex(columns=expected_features, fill_value=0)
    #מעבר ל־NumPy
    X = X_df.astype(np.float32).values
    #normalization
    X = (X - mean) / std
    return X


def extract_attack_rows_and_targets(df):
    #לוקח את ה־labels הגולמיים מה־DataFrame.
    labels = get_raw_labels(df)
    #אם שלחת DataFrame בלי labels, אי אפשר לבנות y
    if labels is None:
        raise ValueError("Attack labels are required for this operation.")
    #יוצר מסכה בוליאנית
    attack_mask = labels != "normal"
    #attack_df = רק השורות של תקיפות
    attack_df = df.loc[attack_mask].copy()
    #attack_labels = רק ה־labels של התקיפות
    attack_labels = labels.loc[attack_mask]
    #אם אין בכלל תקיפות, אין על מה לאמן את המודל השני.
    if attack_df.empty:
        raise ValueError("No attack rows found in the provided dataframe.")
    #מפה את שם ההתקפה הספציפי למשפחת העל שלה
    families = attack_labels.map(ATTACK_FAMILY_MAP)
    unmapped = sorted(set(attack_labels[families.isna()]))
    #אם הלייבל לא קיים במפה הפונקציה עוצרת
    if unmapped:
        raise ValueError(f"Unmapped attack labels found: {unmapped}")
    y = families.map(FAMILY_TO_INDEX).to_numpy(dtype=np.int64)
    return attack_df, y

#זאת פונקציה של חלוקה מאוזנת ל־train ול־validation.
def stratified_split_df(df, y, val_ratio=0.2, seed=42):
    #דיקה בסיסית: לכל שורת קלט חייב להיות label.
    if len(df) != len(y):
        raise ValueError("Features and labels must have the same number of rows.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    #יוצר random generator עם seed קבוע
    rng = np.random.default_rng(seed)
    #מכינים רשימות אינדקסים ל־train ול־val, ומגלים אילו מחלקות קיימות.
    train_positions = []
    val_positions = []
    unique_classes = np.unique(y)
    #לוקח את כל המיקומים של דוגמאות מאותה מחלקה.
    for class_idx in unique_classes:
        class_positions = np.flatnonzero(y == class_idx)
        #מערבב את המיקומים של אותה מחלקה
        class_positions = rng.permutation(class_positions)
        #אנחנו צריכים יותר מדוגמא אחת לפיצול
        #אם יש רק דוגמה אחת במחלקה, אי אפשר לחלק
        if class_positions.size == 1:
            raise ValueError("Each class must contain at least two rows for stratified split.")
        #מחשב כמה דוגמאות מהמחלקה ילכו ל־validation.
        val_size = int(round(class_positions.size * val_ratio))
        val_size = max(1, min(class_positions.size - 1, val_size))
        #עבור כל מחלקה, מחלק את הדוגמאות ל־val ול־train.
        val_positions.extend(class_positions[:val_size].tolist())
        train_positions.extend(class_positions[val_size:].tolist())
    #כדי לא להשאיר את הדוגמאות מסודרות לפי מחלקות
    train_positions = rng.permutation(np.array(train_positions, dtype=np.int64))
    val_positions = rng.permutation(np.array(val_positions, dtype=np.int64))
    #ביצעתי split stratified ידני, כך שכל מחלקה נשמרת בערך באותו יחס ב־train וב־validation. זה חשוב במיוחד בגלל חוסר איזון בין משפחות התקיפה.
    return (
        df.iloc[train_positions].copy(),
        df.iloc[val_positions].copy(),
        y[train_positions],
        y[val_positions],
    )

#ממיר logits לפלט הסתברויות.
def softmax(logits):
    #זה בשביל יציבות נומרית
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

#בוחר את המחלקה עם הציון הכי גבוה
#משווה ל־label האמיתי
#מחזיר ממוצע הצלחות
def multiclass_accuracy(logits, y_true):
    y_pred = np.argmax(logits, axis=1)
    return float(np.mean(y_pred == y_true))


def multiclass_metrics(y_true, logits, label_names=None):
    #אם לא שלחו שמות, משתמשים בברירת המחדל
    if label_names is None:
        label_names = FAMILY_NAMES
    #ממירים logits להסתברויות
    probs = softmax(logits)
    y_pred = np.argmax(probs, axis=1)
    #יצירת confusion matrix
    num_classes = len(label_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[int(true_idx), int(pred_idx)] += 1

    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    #חישוב metrics לכל מחלקה
    for idx, label_name in enumerate(label_names):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        #מתוך כל מה שחזיתי כ־מחלקה X, כמה באמת היו X.
        precision_den = tp + fp
        recall_den = tp + fn
        #מתוך כל מה שבאמת שייך למחלקה X, כמה תפסתי
        precision = float(tp / precision_den) if precision_den > 0 else 0.0
        recall = float(tp / recall_den) if recall_den > 0 else 0.0
        #מוצע הרמוני של precision ו־recall
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        #שמירת per-class report
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "acc": float(np.mean(y_pred == y_true)),
        #Macro metrics נותנים משקל שווה לכל מחלקה, בלי קשר לגודל שלה
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "rows": int(y_true.shape[0]),
        "label_order": list(label_names),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
"""
        בניתי פונקציית evaluation שמפיקה
        confusion matrix, precision, recall ו־F1
        לכל מחלקה
        בנוסף ל־macro averages.
          בחרתי ב־macro F1 
          כי הדאטה לא מאוזן,
          ולכן accuracy לבדה לא מספיקה.
        """

#אוסף את כל מיפויי התוויות למבנה אחד
def save_label_map(path):
    payload = {
        "family_names": FAMILY_NAMES,
        "family_to_index": FAMILY_TO_INDEX,
        "attack_family_map": ATTACK_FAMILY_MAP,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
