import cv2
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_image, val_transforms, CLASSES
from src.model import build_model, get_device
from src.gradcam import get_gradcam_heatmap, overlay_gradcam

MODEL_PATH = "models/skin_detector.pth"


def load_model(device):
    model = build_model(trainable_base=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, img: np.ndarray, device):
    tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
    return probs


def draw_results(frame: np.ndarray, predictions: np.ndarray,
                 heatmap_frame: np.ndarray = None) -> np.ndarray:
    top3_idx = np.argsort(predictions)[-3:][::-1]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, idx in enumerate(top3_idx):
        label = CLASSES[idx]
        conf  = predictions[idx]
        y     = 28 + i * 30

        bar_width = int(conf * 200)
        color = (0, 200, 80) if conf > 0.7 else (0, 180, 230) if conf > 0.4 else (80, 80, 200)
        cv2.rectangle(frame, (10, y + 6), (10 + bar_width, y + 18), color, -1)

        text = f"{label}: {conf:.1%}"
        cv2.putText(frame, text, (10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Solo referencia - consulte a un dermatologo",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    if heatmap_frame is not None:
        h, w = frame.shape[:2]
        small = cv2.resize(heatmap_frame, (w // 3, h // 3))
        frame[h - h//3 - 10 : h - 10, w - w//3 - 10 : w - 10] = small

    return frame


def run_camera():
    device = get_device()
    model  = load_model(device)
    cap    = cv2.VideoCapture(0)

    print("Presiona 'q' para salir | 'c' para capturar y analizar")
    last_predictions = None
    last_heatmap     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx, cy, r = w // 2, h // 2, min(h, w) // 3
        cv2.circle(frame, (cx, cy), r, (0, 230, 130), 2)
        cv2.putText(frame, "Centre la lesion", (cx - 70, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 130), 1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = cv2.resize(rgb, (224, 224))
            preds = predict(model, img, device)

            tensor = val_transforms(img).unsqueeze(0).to(device)
            heatmap = get_gradcam_heatmap(model, tensor)
            last_heatmap = overlay_gradcam(img, heatmap)
            last_heatmap = cv2.cvtColor(last_heatmap, cv2.COLOR_RGB2BGR)
            last_predictions = preds

            top_class = CLASSES[np.argmax(preds)]
            print(f"\nDiagnostico: {top_class} ({preds.max():.1%} confianza)")

        if last_predictions is not None:
            frame = draw_results(frame, last_predictions, last_heatmap)

        cv2.imshow("Detector de Cancer de Piel", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str):
    device = get_device()
    model  = load_model(device)
    img    = preprocess_image(path)
    preds  = predict(model, img, device)

    tensor  = val_transforms(img).unsqueeze(0).to(device)
    heatmap = get_gradcam_heatmap(model, tensor)

    import matplotlib.pyplot as plt
    overlay = overlay_gradcam(img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img);     axes[0].set_title("Original")
    axes[1].imshow(overlay); axes[1].set_title("Grad-CAM")

    sorted_idx = np.argsort(preds)[::-1]
    axes[2].barh([CLASSES[i] for i in sorted_idx], preds[sorted_idx], color="#1D9E75")
    axes[2].set_xlim(0, 1);  axes[2].set_title("Probabilidades")

    plt.tight_layout()
    plt.savefig("resultado.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_image(sys.argv[1])
    else:
        run_camera()