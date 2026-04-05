import numpy as np


class LayerDense:
    """
    Fully-Connected (Dense) Layer:
    Z = XW + b
    """
    def __init__(self, n_inputs: int, n_neurons: int, weight_scale: float = 0.01):
        # weights: (n_inputsכמה פיצרים נכנסים,  n_neuronsכמה נוירונים בשכבה)
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        #יוצרים מטריצה רנדומלית נירונים כפול פצרים מכפילים ב0.1 כדי להתחיל עם משקלות קטנים
        
        
        # biases: (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #ביוס אחד לכול נוירון מתחילים ב0

        # caches
        self.inputs = None
        #שומרים את הקלט כדי להשתמש בו בBACKWORDS

        # gradients
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        #ישמרו אחרי הבאקוורד כדי שהאופטומזירס יתעדכנו

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Save inputs for backward
        self.inputs = inputs
        # Compute output
        output = inputs @ self.weights + self.biases
        #כפל מטריצות
        #מחזירה Z=WX+B
        return output


    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        יוצרים נגזרות על מנת לראות כמה כול שינוי משפיע
        """
        # Gradients w.r.t weights and biases
        self.dweights = self.inputs.T @ dvalues
        #עושים T למטריצה הכוונה להופכית 
        #כדי שהעמודות והשורות יהיו זהים כפי שלמדנו בחוקי מטריצות

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #גרדיאט על הביוס

        # Gradient w.r.t inputs (to pass backward)
        self.dinputs = dvalues @ self.weights.T
        return self.dinputs

#פונקציות אקטביציה
class ActivationReLU:
    """
    ReLU Activation:
    A = max(0, Z)
    """
    def __init__(self):
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        output = np.maximum(0, inputs)
        #כול ערך שלילי נהיה 0
        return output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        #אם בפורוורד יצא ערך שלילי->הוא נחתך ל0 והגראדט ניהיה 0
        return self.dinputs


class ActivationSigmoid:
    """
    Sigmoid Activation:
    A = 1 / (1 + exp(-Z))
    """
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # CLIP-אם ערך קטן מ-500 הוא נהיה -500 וככה גם לצד השני 
        inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs))
        #לפי הנוסחה של סגמונד
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        self.dinputs = dvalues * (self.output * (1 - self.output))
        #נגזרת של סיגמונד,אם הסיגמונד קרוב ל0,1 הנגזרת קטנה
        return self.dinputs


class ActivationSoftmax:
    """
    Softmax Activation (for multiclass):
    P_i = exp(z_i) / sum_j exp(z_j)
    """
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max per row
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        #axis=1 מקסימום בכול שורה . keepdims=TRUE שומר את הצורה

        exp_values = np.exp(shifted)
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        #מחסרים מקסימום בכול שורה כדי למנוע אוברפלו
        #בסוף מקבלים הסתברות שהסכום 1 לכול דוגמא
        self.output = probs
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        General softmax backward is expensive.
        In practice, we combine Softmax + CrossEntropy loss for a simpler gradient.
        But this is a correct generic version (O(batch * classes^2)).
        """
        self.dinputs = np.zeros_like(dvalues)

        for i, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)  # (classes, 1)
            jacobian = np.diagflat(single_output) - (single_output @ single_output.T)
            self.dinputs[i] = (jacobian @ single_dvalue).reshape(-1)

        return self.dinputs
