
import sounddevice as sd
import numpy as np

DEVICE = 1
SAMPLE_RATE = 44100
DURATION = 10

print("Testing microphone:", sd.query_devices(DEVICE)["name"])
print("Speak normally for 10 seconds...")

try:
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=DEVICE,
    )
    sd.wait()

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    print(f"Peak audio level: {peak:.5f}")
    print(f"Average audio level: {rms:.5f}")

    if peak < 0.001:
        print("No meaningful audio detected. Check Windows microphone settings.")
    else:
        print("Microphone is capturing audio successfully.")

except Exception as exc:
    print("Microphone error:", exc)
