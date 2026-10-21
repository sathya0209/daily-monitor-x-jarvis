
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
model_files = list(MODEL_DIR.glob("*.onnx"))

if not model_files:
    raise FileNotFoundError(
        "No ONNX model found in the models folder."
    )

model_path = model_files[0]
print(f"Loading model: {model_path.name}")

model = Model(
    wakeword_models=[str(model_path)],
    inference_framework="onnx",
)

DEVICE = 1
INPUT_RATE = 44100
MODEL_RATE = 16000
INPUT_BLOCK = 4410
FRAME_SIZE = 1280
THRESHOLD = 0.30

buffer = np.array([], dtype=np.int16)
last_detection = 0.0
frame_count = 0

print(f"Microphone: {sd.query_devices(DEVICE)['name']}")
print("Speak normally, then say: Hey Jarvis")
print("Press Ctrl+C to stop.\n")

try:
    with sd.InputStream(
        device=DEVICE,
        samplerate=INPUT_RATE,
        channels=1,
        dtype="int16",
        blocksize=INPUT_BLOCK,
    ) as stream:

        while True:
            audio, overflowed = stream.read(INPUT_BLOCK)
            audio = np.asarray(audio, dtype=np.int16).reshape(-1)

            # Convert 44.1 kHz microphone audio to 16 kHz.
            converted = resample_poly(audio, 160, 441)
            converted = np.clip(
                converted, -32768, 32767
            ).astype(np.int16)

            buffer = np.concatenate((buffer, converted))

            while len(buffer) >= FRAME_SIZE:
                frame = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]
                frame_count += 1

                rms = float(
                    np.sqrt(
                        np.mean(frame.astype(np.float32) ** 2)
                    )
                )

                scores = model.predict(frame)
                best_name = max(
                    scores, key=scores.get
                ) if scores else "no score"
                best_score = (
                    float(scores[best_name])
                    if scores else 0.0
                )

                if frame_count % 10 == 0:
                    print(
                        f"Mic RMS={rms:.0f} | "
                        f"{best_name}={best_score:.3f}"
                        + (" | overflow" if overflowed else "")
                    )

                if (
                    best_score >= THRESHOLD
                    and time.monotonic() - last_detection >= 2
                ):
                    print(
                        f"\nWake-word candidate: {best_name} "
                        f"(score={best_score:.3f})\n"
                    )
                    last_detection = time.monotonic()
                    model.reset()

except KeyboardInterrupt:
    print("\nTest stopped.")
except Exception as exc:
    print(f"\nTest error: {type(exc).__name__}: {exc}")
