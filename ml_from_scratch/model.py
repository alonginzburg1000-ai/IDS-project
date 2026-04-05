import numpy as np


class DFFNN:
    """
    Simple feed-forward neural network (from scratch).
    Holds layers in a list. Each layer must implement:
      - forward(inputs) -> outputs
      - backward(dvalues) -> dinputs
    Dense layers also have:
      - weights, biases, dweights, dbiases
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Pass input through each layer sequentially.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
        """
        1.מתחילים מקלט אקס
        2.מעבירים לשכבה הראשונה ומקבלים פלט
        3.את הפלט מעבירים לשכבה הבאה וכו
        4.בסוף מחזירים את הפלט(תחזית)
        """

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        Backpropagate gradients in reverse order.
        """
        grad = dvalues
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        """
        מתחילים מהגאדינט שמגיע מLOSS
        הולכים שכבה שכבה בסדר הפוך
        כול שכבה
        *מחשבת את הגרידאנטים
        *מחזיר DINPUT לשכבה הקודמת
        """

    def update(self, lr: float):
        """
        SGD update for Dense layers only.
        If a layer has weights/biases and dweights/dbiases, update them.
        """
        for layer in self.layers:
            #HASATTR - בודקת אם יש לאוביקט תכונה מסויימת
            if hasattr(layer, "weights") and hasattr(layer, "dweights"):
                layer.weights -= lr * layer.dweights
                layer.biases -= lr * layer.dbiases
    #מבצעת אופטמיזר פשוט
    """
        מה זה עושה-עובר על  השכבות
        רק לשכבות DENSE באמת יש משקולות וביוס 
        """
    def _dense_layers(self):
        """
        Return only layers that hold trainable dense parameters.
        """
        return [layer for layer in self.layers if hasattr(layer, "weights") and hasattr(layer, "biases")]

    def save_weights(self, path: str):
        """
        Save Dense layer weights and biases into a single .npz file.
        """
        payload = {}
        for i, layer in enumerate(self._dense_layers()):
            payload[f"W{i}"] = layer.weights
            payload[f"b{i}"] = layer.biases
        np.savez(path, **payload)

    def load_weights(self, path: str):
        """
        Load Dense layer weights and biases from a .npz file.
        """
        data = np.load(path)
        dense_layers = self._dense_layers()
        for i, layer in enumerate(dense_layers):
            layer.weights = data[f"W{i}"]
            layer.biases = data[f"b{i}"]
