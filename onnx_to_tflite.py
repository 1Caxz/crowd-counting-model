import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("csrnet_mobile.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("csrnet_mobile_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("csrnet_mobile_tf")
tflite_model = converter.convert()

with open("csrnet_mobile.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TFLite: csrnet_mobile.tflite")
