import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("simple_net.onnx")

dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

outputs = ort_session.run(None, {"input": dummy_input})
print("Model output:", outputs)

