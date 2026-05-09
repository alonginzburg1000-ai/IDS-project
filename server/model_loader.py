from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from ml_from_scratch.layers import ActivationReLU, ActivationSigmoid, LayerDense
from ml_from_scratch.model import DFFNN


@dataclass(frozen=True)
class PreprocessState:
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]


@dataclass(frozen=True)
class RuntimeModels:
    binary_model: DFFNN
    binary_preprocess: PreprocessState
    multiclass_model: DFFNN
    multiclass_preprocess: PreprocessState
    family_names: List[str]


def load_runtime_models(artifacts_path: Path) -> RuntimeModels:
    _require_file(artifacts_path / "binary_model_weights_best.npz")
    _require_file(artifacts_path / "multiclass_model_weights_best.npz")
    _require_file(artifacts_path / "binary_preprocess.npz")
    _require_file(artifacts_path / "multiclass_preprocess.npz")
    _require_file(artifacts_path / "multiclass_label_map.json")

    binary_preprocess = _load_preprocess(artifacts_path / "binary_preprocess.npz")
    multiclass_preprocess = _load_preprocess(artifacts_path / "multiclass_preprocess.npz")
    binary_model = _build_binary_model_from_weights(artifacts_path / "binary_model_weights_best.npz")
    multiclass_model = _build_multiclass_model_from_weights(
        artifacts_path / "multiclass_model_weights_best.npz"
    )

    if len(binary_preprocess.feature_names) != _first_weight_input_size(
        artifacts_path / "binary_model_weights_best.npz"
    ):
        raise ValueError("Binary feature count does not match binary model input size.")
    if len(multiclass_preprocess.feature_names) != _first_weight_input_size(
        artifacts_path / "multiclass_model_weights_best.npz"
    ):
        raise ValueError("Multiclass feature count does not match multiclass model input size.")

    family_names = _load_family_names(artifacts_path / "multiclass_label_map.json")
    multiclass_output_size = _last_weight_output_size(artifacts_path / "multiclass_model_weights_best.npz")
    if len(family_names) != multiclass_output_size:
        raise ValueError(
            "Label map has {} classes, but multiclass model outputs {}.".format(
                len(family_names), multiclass_output_size
            )
        )

    return RuntimeModels(
        binary_model=binary_model,
        binary_preprocess=binary_preprocess,
        multiclass_model=multiclass_model,
        multiclass_preprocess=multiclass_preprocess,
        family_names=family_names,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _load_preprocess(path: Path) -> PreprocessState:
    with np.load(str(path), allow_pickle=True) as pre:
        mean = pre["mean"].astype(np.float32)
        std = pre["std"].astype(np.float32)
        feature_names = [str(name) for name in pre["feature_names"].tolist()]
    return PreprocessState(mean=mean, std=std, feature_names=feature_names)


def _build_binary_model_from_weights(weights_path: Path) -> DFFNN:
    n_inputs, hidden1, hidden2, output_size = _dense_shapes(weights_path)
    if output_size != 1:
        raise ValueError("Binary model must have output size 1, got {}.".format(output_size))
    model = DFFNN(
        [
            LayerDense(n_inputs=n_inputs, n_neurons=hidden1),
            ActivationReLU(),
            LayerDense(n_inputs=hidden1, n_neurons=hidden2),
            ActivationReLU(),
            LayerDense(n_inputs=hidden2, n_neurons=1),
            ActivationSigmoid(),
        ]
    )
    model.load_weights(str(weights_path))
    return model


def _build_multiclass_model_from_weights(weights_path: Path) -> DFFNN:
    n_inputs, hidden1, hidden2, output_size = _dense_shapes(weights_path)
    model = DFFNN(
        [
            LayerDense(n_inputs=n_inputs, n_neurons=hidden1),
            ActivationReLU(),
            LayerDense(n_inputs=hidden1, n_neurons=hidden2),
            ActivationReLU(),
            LayerDense(n_inputs=hidden2, n_neurons=output_size),
        ]
    )
    model.load_weights(str(weights_path))
    return model


def _dense_shapes(weights_path: Path) -> Tuple[int, int, int, int]:
    with np.load(str(weights_path)) as weights:
        _require_weight_keys(weights)
        n_inputs = int(weights["W0"].shape[0])
        hidden1 = int(weights["W0"].shape[1])
        hidden2 = int(weights["W1"].shape[1])
        output_size = int(weights["W2"].shape[1])
    return n_inputs, hidden1, hidden2, output_size


def _first_weight_input_size(weights_path: Path) -> int:
    with np.load(str(weights_path)) as weights:
        return int(weights["W0"].shape[0])


def _last_weight_output_size(weights_path: Path) -> int:
    with np.load(str(weights_path)) as weights:
        return int(weights["W2"].shape[1])


def _require_weight_keys(weights: Any) -> None:
    missing = [key for key in ("W0", "b0", "W1", "b1", "W2", "b2") if key not in weights.files]
    if missing:
        raise ValueError("Missing weight arrays: {}".format(", ".join(missing)))


def _load_family_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    family_names = payload.get("family_names")
    if isinstance(family_names, list) and family_names:
        return [str(name) for name in family_names]

    family_to_index = payload.get("family_to_index")
    if isinstance(family_to_index, dict) and family_to_index:
        indexed = sorted((int(index), str(name)) for name, index in family_to_index.items())
        return [name for _, name in indexed]

    raise ValueError("multiclass_label_map.json must contain family_names or family_to_index.")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError("Missing required artifact: {}".format(path))
