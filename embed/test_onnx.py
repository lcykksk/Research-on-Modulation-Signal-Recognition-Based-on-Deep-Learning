import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("embed/HANet_IQ_LSTM.onnx")

print("Inputs:", session.get_inputs())
print("Outputs:", session.get_outputs())

x = np.random.randn(1, 2, 128, 1).astype(np.float32)
y = session.run(None, {"input": x})

print("Output shape:", y[0].shape)