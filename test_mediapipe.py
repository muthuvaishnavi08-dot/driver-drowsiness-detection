import sys
print("Python:", sys.version)

# Try all possible import methods
methods = [
    ("import mediapipe; mediapipe.face_mesh", "import mediapipe; print(mediapipe.face_mesh)"),
    ("mediapipe solutions", "from mediapipe import solutions; print(solutions.face_mesh)"),
    ("mediapipe.solutions.face_mesh", "from mediapipe.solutions import face_mesh; print(face_mesh)"),
]

for name, code in methods:
    try:
        exec(code)
        print(f"✓ SUCCESS: {name}")
        break
    except Exception as e:
        print(f"✗ FAILED: {name} -> {e}")