import importlib.metadata
import sys

# รายการไลบรารีและเวอร์ชันที่คาดหวัง (Requirements)
requirements = {
    "tensorflow": "2.15.0",
    "keras-tuner": "1.4.7",  # >= 1.4.7
    "mediapipe": "0.10.14",
    "opencv-python": "4.9.0.80", # >= 4.9.0.80
    "numpy": "1.x",          # < 2.0.0 (Cheating check by logic below)
    "pandas": "2.2.0",       # >= 2.2.0
    "scikit-learn": "1.4.0", # >= 1.4.0
    "tqdm": "4.66.1",        # >= 4.66.1
    "matplotlib": "3.8.2"    # >= 3.8.2
}

print(f"{'LIBRARY':<20} | {'INSTALLED':<15} | {'STATUS':<10}")
print("-" * 50)

all_pass = True

for lib, req_ver in requirements.items():
    try:
        # ดึงเวอร์ชันที่ติดตั้งอยู่
        installed_ver = importlib.metadata.version(lib)
        
        status = "✅ OK"
        
        # Logic ตรวจสอบพิเศษ
        if lib == "numpy":
            # ต้องน้อยกว่า 2.0.0
            if installed_ver.startswith("2."):
                status = "❌ TOO NEW (<2.0.0)"
                all_pass = False
        elif lib == "tensorflow":
            # ต้องตรงเป๊ะ (แนะนำ)
            if installed_ver != req_ver:
                status = f"⚠️ Diff ({req_ver})"
        
        print(f"{lib:<20} | {installed_ver:<15}")
        
    except importlib.metadata.PackageNotFoundError:
        print(f"{lib:<20} | {'Not Found':<15} | ❌ MISSING")
        all_pass = False

print("-" * 50)

# เช็ค Python Version ด้วย (แถมให้)
py_ver = sys.version.split()[0]
print(f"{'Python':<20} | {py_ver:<15}")

print("-" * 50)
if all_pass:
    print("🎉 Environment พร้อมใช้งาน 100% ครับ!")
else:
    print("⚠️ พบปัญหาบางอย่าง (ดูเครื่องหมาย ❌ หรือ ⚠️)")