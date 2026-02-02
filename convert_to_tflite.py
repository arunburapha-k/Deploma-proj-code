import os, tensorflow as tf
MODEL_DIR   = "models"
KERAS_MODEL = os.path.join(MODEL_DIR, "best_model.keras")

OUT_FP32 = os.path.join(MODEL_DIR, "model_fp32.tflite")
OUT_FP16 = os.path.join(MODEL_DIR, "model_fp16.tflite")


print("TF:", tf.__version__)
tf.keras.backend.clear_session()

# ---- โหลดโมเดล ----
model = tf.keras.models.load_model(KERAS_MODEL)

# ใช้ concrete function เพื่อ fix shape และ freeze graph (กัน TensorList/variable ค้าง)
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 30, 258], dtype=tf.float32, name='input')
])
def serving(x):
    return model(x, training=False)

concrete = serving.get_concrete_function()

def dump_ops(tflite_path):
    # ตรวจชื่อ op หลังแปลงเพื่อให้มั่นใจว่าไม่มี SELECT_TF_OPS
    import flatbuffers
    from tflite_support import flatbuffers as fbs  # ถ้าไม่มี ให้ข้ามได้
    try:
        from tflite import Model
    except Exception:
        print("[info] skip op dump (schema package not installed)")
        return
    buf = open(tflite_path, "rb").read()
    m = Model.Model.GetRootAsModel(buf, 0)
    ops = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        ops.append(oc.BuiltinCode())
    print(f"[{os.path.basename(tflite_path)}] builtin op codes:", ops)

# ---- FP32 (ไม่มี quantization) ----
conv_fp32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
conv_fp32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# ไม่ตั้ง optimizations = ไม่มี quantization
tfl_fp32 = conv_fp32.convert()
open(OUT_FP32, "wb").write(tfl_fp32)
print("Saved:", OUT_FP32)

# ---- FP16 (weights เป็น float16, I/O ยัง float32) ----
conv_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
conv_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_fp16.target_spec.supported_types = [tf.float16]
tfl_fp16 = conv_fp16.convert()
open(OUT_FP16, "wb").write(tfl_fp16)
print("Saved:", OUT_FP16)

# (ออปชัน) ตรวจว่าโหลดได้จริง
for p in [OUT_FP32, OUT_FP16]:
    interp = tf.lite.Interpreter(model_path=p)
    interp.allocate_tensors()
    idef = interp.get_input_details()[0]
    odef = interp.get_output_details()[0]
    print(f"[OK] {os.path.basename(p)} input={idef['shape']},{idef['dtype']}  output={odef['shape']},{odef['dtype']}")
