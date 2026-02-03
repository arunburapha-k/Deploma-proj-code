import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

MODEL_DIR   = "models"
KERAS_MODEL = os.path.join(MODEL_DIR, "best_model.keras")

OUT_FP32 = os.path.join(MODEL_DIR, "model_fp32.tflite")
OUT_FP16 = os.path.join(MODEL_DIR, "model_fp16.tflite")

print("TF:", tf.__version__)
tf.keras.backend.clear_session()

# ----------------- 1. ใส่ Class Attention ให้เหมือนไฟล์เทรนเป๊ะๆ -----------------
# (ต้องมีสิ่งนี้ ไม่งั้นโหลดโมเดลไม่ได้)
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# ----------------- 2. โหลดโมเดลพร้อม Custom Object -----------------
print("Loading model...")
model = tf.keras.models.load_model(
    KERAS_MODEL,
    custom_objects={'Attention': Attention} # <--- บอกให้รู้ว่า Attention คือ class ข้างบนนี้
)
print("Model loaded successfully!")

# ----------------- 3. เตรียมฟังก์ชัน Serving (Fix Shape) -----------------
# ใช้ concrete function เพื่อ fix shape และ freeze graph (กัน TensorList/variable ค้าง)
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 30, 258], dtype=tf.float32, name='input')
])
def serving(x):
    return model(x, training=False)

concrete = serving.get_concrete_function()

# ----------------- 4. แปลงเป็น TFLite (FP32) -----------------
print("Converting to FP32...")
conv_fp32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
conv_fp32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl_fp32 = conv_fp32.convert()

with open(OUT_FP32, "wb") as f:
    f.write(tfl_fp32)
print("Saved:", OUT_FP32)

# ----------------- 5. แปลงเป็น TFLite (FP16) -----------------
print("Converting to FP16...")
conv_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
conv_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_fp16.target_spec.supported_types = [tf.float16]
tfl_fp16 = conv_fp16.convert()

with open(OUT_FP16, "wb") as f:
    f.write(tfl_fp16)
print("Saved:", OUT_FP16)

# ----------------- 6. ตรวจสอบไฟล์ผลลัพธ์ -----------------
for p in [OUT_FP32, OUT_FP16]:
    if not os.path.exists(p): continue
    try:
        interp = tf.lite.Interpreter(model_path=p)
        interp.allocate_tensors()
        idef = interp.get_input_details()[0]
        odef = interp.get_output_details()[0]
        print(f"[OK] {os.path.basename(p)} input={idef['shape']},{idef['dtype']}  output={odef['shape']},{odef['dtype']}")
    except Exception as e:
        print(f"[ERROR] Checking {os.path.basename(p)} failed: {e}")