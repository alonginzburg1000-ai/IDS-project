from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from server.model_loader import RuntimeModels, softmax
from server.preprocessing import packet_json_to_normalized_vector


class InferenceEngine:
    def __init__(self, models: RuntimeModels, binary_threshold: float = 0.55):
        self.models = models
        self.binary_threshold = binary_threshold

    def predict(self, packet_payload: Mapping[str, Any]) -> Dict[str, Any]:
        binary_x = packet_json_to_normalized_vector(
            packet_payload,
            self.models.binary_preprocess.feature_names,
            self.models.binary_preprocess.mean,
            self.models.binary_preprocess.std,
        )
        binary_attack_probability = float(self.models.binary_model.forward(binary_x).reshape(-1)[0])

        is_attack = binary_attack_probability >= self.binary_threshold
        if not is_attack:
            return {
                "binary_prediction": "normal",
                "binary_confidence": float(1.0 - binary_attack_probability),
                "attack_type": None,
                "multiclass_confidence": None,
            }

        multiclass_x = packet_json_to_normalized_vector(
            packet_payload,
            self.models.multiclass_preprocess.feature_names,
            self.models.multiclass_preprocess.mean,
            self.models.multiclass_preprocess.std,
        )
        logits = self.models.multiclass_model.forward(multiclass_x)
        probabilities = softmax(logits)
        predicted_index = int(np.argmax(probabilities, axis=1)[0])
        attack_type = self.models.family_names[predicted_index]
        multiclass_confidence = float(probabilities[0, predicted_index])

        return {
            "binary_prediction": "attack",
            "binary_confidence": binary_attack_probability,
            "attack_type": attack_type,
            "multiclass_confidence": multiclass_confidence,
        }
