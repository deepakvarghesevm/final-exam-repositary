from __future__ import annotations

import math
import os
import socket
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "adult_income_pipeline.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME

app = Flask(__name__)


def load_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        app.logger.exception("The saved model could not be loaded")
        raise RuntimeError("The model could not be loaded.")


model = load_model()


def get_feature_schema() -> dict[str, dict[str, Any]]:
    feature_names = list(getattr(model, "feature_names_in_", []))
    numeric_features: set[str] = set()
    categorical_features: dict[str, list[Any]] = {}

    preprocessor = None
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")

    if preprocessor is not None:
        transformers = getattr(preprocessor, "transformers", [])
        fitted_transformers = getattr(preprocessor, "named_transformers_", {})
        for name, transformer, columns in transformers:
            if name == "remainder":
                continue
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                column_names = [str(column) for column in columns]
            else:
                column_names = [str(columns)]

            if name == "num" or "numeric" in str(name).lower():
                numeric_features.update(column_names)
            fitted_transformer = fitted_transformers.get(name, transformer)
            if hasattr(fitted_transformer, "categories_"):
                for column, categories in zip(column_names, fitted_transformer.categories_):
                    categorical_features[column] = [str(value) for value in categories]

    for name in feature_names:
        lower_name = name.lower()
        if lower_name in {"is_standalone_entity", "is_standalone", "standalone"}:
            categorical_features[name] = ["True", "False"]
        elif name not in numeric_features and name not in categorical_features:
            if any(token in lower_name for token in ("count", "year", "depth", "employee")):
                numeric_features.add(name)

    schema: dict[str, dict[str, Any]] = {}
    for name in feature_names:
        if name in categorical_features:
            schema[name] = {"type": "categorical", "choices": categorical_features[name]}
        elif name in numeric_features:
            integer_like = any(
                token in name.lower()
                for token in ("count", "year", "depth", "employee")
            )
            schema[name] = {"type": "integer" if integer_like else "number"}
        else:
            schema[name] = {"type": "text"}
    return schema


FEATURES = get_feature_schema()
FEATURE_ORDER = list(FEATURES)


def find_available_port(preferred_port: int) -> int:
    if preferred_port == 0:
        return 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            for port in range(preferred_port + 1, preferred_port + 100):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
                    try:
                        candidate.bind(("127.0.0.1", port))
                        return port
                    except OSError:
                        continue
    raise RuntimeError("No available local port was found.")


def parse_payload(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Please submit the form values as valid fields.")

    missing = [name for name in FEATURE_ORDER if name not in payload or payload[name] == ""]
    if missing:
        raise ValueError("Please complete every required field.")

    row: dict[str, Any] = {}
    for name in FEATURE_ORDER:
        raw_value = payload[name]
        metadata = FEATURES[name]
        feature_type = metadata["type"]

        if feature_type == "categorical":
            text_value = str(raw_value).strip()
            if name.lower() in {"is_standalone_entity", "is_standalone", "standalone"}:
                lowered = text_value.lower()
                if lowered not in {"true", "false", "1", "0"}:
                    raise ValueError("Please select a valid standalone entity value.")
                row[name] = lowered in {"true", "1"}
            else:
                if text_value not in metadata["choices"]:
                    raise ValueError("Please select a valid value.")
                row[name] = text_value
        elif feature_type in {"integer", "number"}:
            try:
                number = float(str(raw_value).strip())
            except (TypeError, ValueError):
                raise ValueError("Please enter valid numeric values.")
            if not math.isfinite(number):
                raise ValueError("Please enter valid numeric values.")
            if feature_type == "integer":
                if not number.is_integer():
                    raise ValueError("Please enter whole numbers where required.")
                row[name] = int(number)
            else:
                row[name] = number
        else:
            text_value = str(raw_value).strip()
            if not text_value:
                raise ValueError("Please complete every required field.")
            row[name] = text_value

    return pd.DataFrame([row], columns=FEATURE_ORDER)


def format_prediction(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value[0]
    try:
        number = float(value)
        if math.isfinite(number):
            return f"{number:,.2f}"
    except (TypeError, ValueError):
        pass
    return str(value)


def predict_input(input_frame: pd.DataFrame) -> tuple[bool, str, list[dict[str, Any]] | None]:
    is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")
    prediction = model.predict(input_frame)
    prediction_value = prediction[0]

    probabilities: list[dict[str, Any]] | None = None
    if is_classifier and hasattr(model, "predict_proba"):
        probability_row = model.predict_proba(input_frame)[0]
        class_values = getattr(model, "classes_", None)
        if class_values is None:
            final_estimator = getattr(model, "steps", [])
            for _, estimator in reversed(final_estimator):
                if hasattr(estimator, "classes_"):
                    class_values = estimator.classes_
                    break
        if class_values is not None:
            probabilities = [
                {"label": str(label), "probability": float(probability)}
                for label, probability in zip(class_values, probability_row)
            ]

    if is_classifier:
        return True, str(prediction_value), probabilities
    return False, format_prediction(prediction_value), None


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_filename=MODEL_FILENAME,
        feature_order=FEATURE_ORDER,
        features=FEATURES,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "model": MODEL_FILENAME})


@app.post("/predict")
def predict():
    try:
        if request.is_json:
            payload = request.get_json(silent=True)
        else:
            payload = request.form.to_dict()
        input_frame = parse_payload(payload or {})
        is_classifier, label, probabilities = predict_input(input_frame)
        return jsonify(
            {
                "success": True,
                "prediction": label,
                "is_classifier": is_classifier,
                "probabilities": probabilities,
            }
        )
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(
            {"success": False, "error": "Prediction could not be completed. Please check your values and try again."}
        ), 400


if __name__ == "__main__":
    preferred_port = int(os.getenv("FLASK_PORT", "5000"))
    port = find_available_port(preferred_port)
    print(f"Model: {MODEL_FILENAME}")
    print(f"Open: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
