#!/usr/bin/env python3
"""
Fundus Gate Inference CLI
Validates whether an input image is a Retinal Fundus Photograph before passing to DR classifier.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model fundus_gate.tflite --threshold 0.63
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

def find_default_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_threshold(threshold_arg, script_dir):
    if threshold_arg is not None:
        return float(threshold_arg)
    
    # Check threshold.json in script dir or current dir
    candidates = [
        os.path.join(script_dir, "threshold.json"),
        os.path.join(script_dir, "output", "threshold.json"),
        "threshold.json"
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return float(data.get("optimal_threshold", 0.5))
            except Exception:
                pass
    return 0.5

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        sys.exit(f"Error opening image '{image_path}': {e}")
    
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    
    # MobileNetV3 internal architecture includes a Rescaling(1/255.) layer,
    # so the model expects raw RGB float32 pixels in range [0, 255].
    img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    return img_array

def run_tflite_inference(model_path, input_tensor):
    interpreter = None
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=model_path)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
            except ImportError:
                sys.exit("Error: None of 'ai_edge_litert', 'tflite_runtime', or 'tensorflow' is installed. Please install one of them.")
            
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Ensure dtype matches
    input_dtype = input_details[0]['dtype']
    if input_tensor.dtype != input_dtype:
        input_tensor = input_tensor.astype(input_dtype)
        
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return float(output_data.flatten()[0])

def run_keras_inference(model_path, input_tensor):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        preds = model.predict(input_tensor, verbose=0)
        return float(preds.flatten()[0])
    except ImportError:
        sys.exit("Error: 'tensorflow' is required to load .keras model.")

def main():
    parser = argparse.ArgumentParser(description="Pre-screening Fundus Gate for DR Classifier")
    parser.add_argument("--image", required=True, type=str, help="Path to input image file")
    parser.add_argument("--model", type=str, default=None, help="Path to model (.tflite or .keras)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Resolve model path
    model_path = args.model
    if model_path is None:
        model_candidates = [
            os.path.join(script_dir, "fundus_gate_float16.tflite"),
            os.path.join(script_dir, "fundus_gate.tflite"),
            os.path.join(script_dir, "fundus_gate.keras"),
            os.path.join(script_dir, "output", "fundus_gate_float16.tflite"),
            os.path.join(script_dir, "output", "fundus_gate.tflite"),
            os.path.join(script_dir, "output", "fundus_gate.keras"),
            "fundus_gate.tflite",
            "fundus_gate.keras"
        ]
        model_path = find_default_file(model_candidates)
        if model_path is None:
            sys.exit("Error: No model found. Please specify --model path/to/model.")

    # 2. Resolve threshold
    threshold = load_threshold(args.threshold, script_dir)

    # 3. Preprocess Image
    if not os.path.exists(args.image):
        sys.exit(f"Error: Image file not found: {args.image}")
    input_tensor = preprocess_image(args.image)

    # 4. Predict
    if model_path.endswith(".tflite"):
        prob = run_tflite_inference(model_path, input_tensor)
    elif model_path.endswith(".keras") or model_path.endswith(".h5"):
        prob = run_keras_inference(model_path, input_tensor)
    else:
        sys.exit(f"Unsupported model extension: {model_path}")

    decision = "ACCEPT" if prob >= threshold else "REJECT"

    # Exact required CLI output format
    print(f"Probability: {prob:.2f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Decision: {decision}")

if __name__ == "__main__":
    main()
