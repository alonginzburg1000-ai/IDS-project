import numpy as np


class BinaryCrossEntropy:
    """
    LOSS
    פונקציה שמקבלת
    -תחזית המודל Y_PRED
    -אמת
    ואז היא עושה BACKWORD -> להחזיר נגזרת גראדינט שיאפשר ללמד את הרשת
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.dinputs = None
        """
        EPS-> מספר קטן שלא יהיה ב LOG(0)
        DINPUTS-> ישמור את הגראדינט שיחזור אחורה
         """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        # כדי למנוע מצב ש Y_PRED יהיה 0 או אחד בדיוק

        # make sure shapes are consistent(ישור צורות)
        y_true = y_true.reshape(y_pred.shape)

        # BCE (נוסחה): -( y*log(p) + (1-y)*log(1-p) )
        sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        """
       אם Y_TRUE=1 -> נשארנו LOG(PRED) כלומר אם P גדול
       יהיה עונש קטן 

        אם Y_TRUE=0 -> נשארנו LOG(1-PRED) ->כלומר ההפך P קטן 
        עונש גדול
        """
        # return mean loss
        return float(np.mean(sample_losses))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        פה אנחנו מחזירים DP
        כמה ההפסד ישתנה אם נשנה את ההסתברות P
        """
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        y_true = y_true.reshape(y_pred.shape)

        batch_size = y_pred.shape[0]

        self.dinputs = (-(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / batch_size
        return self.dinputs


class SoftmaxCrossEntropy:
    """
    Combined Softmax + Categorical Cross-Entropy.
    This is the standard for multi-class classification.

    Input: logits (raw scores) shape (batch, num_classes)
    y_true can be:
      - integer labels shape (batch,)
      - one-hot labels shape (batch, num_classes)

    forward(logits, y_true) -> scalar loss
    backward(logits, y_true) -> dL/dlogits
    """

    def __init__(self):
        self.probs = None
        self.dinputs = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        # logits->מספרים גולמים שיוצאים מהשכבה האחרונה לפני סופטמקס,הם יכולים להיות כול מספר
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.probs = probs
        #שומרים בסלף פרוב כי הוא צריך אותם לBACKWORDS

        batch_size = logits.shape[0]

        # If y_true is one-hot, convert to class indices
        if y_true.ndim == 2:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        # Cross-entropy loss: -log(p_correct)
        correct_confidences = probs[np.arange(batch_size), y_true_idx]
        loss = -np.mean(np.log(correct_confidences + 1e-12))
        return float(loss)

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradient for combined Softmax+CE:
        dL/dlogits = (probs - y_onehot) / batch
        """
        batch_size = logits.shape[0]

        # If y_true is one-hot, convert to class indices
        if y_true.ndim == 2:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        dlogits = self.probs.copy()
        dlogits[np.arange(batch_size), y_true_idx] -= 1
        dlogits /= batch_size
        """
        לכול מחלקה->"כמה עודף נתתי לה
        למחלקה הנכונה אנחנו מורידים 1 כי רצינו 1 שם ואז מחלקים בבאץ
        """
        self.dinputs = dlogits
        return self.dinputs
