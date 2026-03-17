import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import tf2onnx
from HANet_ablation.HANet_ablate_only_IQ import HANet_only_IQ

model = HANet_only_IQ()
model.load_weights("results_ablate/HANet_only_IQ/weights.keras")

spec = (tf.TensorSpec((None, 2, 128, 1), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="embed/HANet_only_IQ.onnx"
)

print("Export done.")

