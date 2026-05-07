
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Standalone model — no DeepfakeBench registry dependency
from recce_model import Recce

CHECKPOINT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..',
    'DeepfakeBench', 'logs', 'training',
    'recce_2026-03-24-13-11-58', 'test', 'avg', 'ckpt_best.pth'
))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same normalization used during training (from recce.yaml)
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_model = None
_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _face_detector


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _load_model():
    model = Recce(num_classes=2, drop_rate=0.2)
    print(f'[inference] Loading checkpoint: {CHECKPOINT_PATH}')
    print(f'[inference] Device: {DEVICE}')

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Checkpoint was saved as RecceDetector.state_dict().
    # The Recce model lives under the "model.*" prefix inside that dict.
    model_state = {k[len('model.'):]: v for k, v in ckpt.items() if k.startswith('model.')}

    if not model_state:
        # Fallback: checkpoint may already be a bare Recce state dict
        model_state = ckpt

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f'[inference] Missing keys (first 5): {missing[:5]}')

    model.to(DEVICE)
    model.eval()
    print('[inference] Model ready.')
    return model


def _crop_face(frame_bgr):
    """Detect and crop the largest face in a BGR frame.

    Returns (rgb_crop, face_was_detected).
    Falls back to a square center crop when no face is found.
    """
    detector = _get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if len(faces) > 0:
        # Take the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(0.25 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(rgb.shape[1], x + w + margin)
        y2 = min(rgb.shape[0], y + h + margin)
        return rgb[y1:y2, x1:x2], True

    # No face found — use square center crop
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s], False


def predict_video(video_path, num_frames=8):
    """Run RECCE inference on a video file.

    Returns a dict with:
        verdict        – 'FAKE' or 'REAL'
        confidence     – 0-100, confidence in the verdict
        fake_prob      – 0-100, average fake probability
        frame_probs    – list of per-frame fake probabilities (0-100)
        frames_analyzed– number of frames processed
        face_detected  – whether a face was found in any frame
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError('Could not read video or video has no frames.')

    sample_n = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_n, dtype=int)

    crops = []
    any_face = False

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        crop, detected = _crop_face(frame)
        crops.append(crop)
        if detected:
            any_face = True

    cap.release()

    if not crops:
        raise ValueError('No frames could be extracted from the video.')

    model = get_model()
    probs = []

    with torch.no_grad():
        for crop in crops:
            img = Image.fromarray(crop)
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            embedding = model.features(tensor)
            logits = model.classifier(embedding)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)

    avg_prob = float(np.mean(probs))
    verdict = 'FAKE' if avg_prob >= 0.5 else 'REAL'
    confidence = avg_prob * 100 if verdict == 'FAKE' else (1.0 - avg_prob) * 100

    return {
        'verdict': verdict,
        'confidence': round(confidence, 1),
        'fake_prob': round(avg_prob * 100, 1),
        'frame_probs': [round(p * 100, 1) for p in probs],
        'frames_analyzed': len(probs),
        'face_detected': any_face,
    }
