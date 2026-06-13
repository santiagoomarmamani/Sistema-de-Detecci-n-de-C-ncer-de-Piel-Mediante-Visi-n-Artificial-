import numpy as np
import cv2
import torch
import torch.nn.functional as F


def get_gradcam_heatmap(model, img_tensor: torch.Tensor) -> np.ndarray:
    """Genera un heatmap Grad-CAM sobre la ultima capa convolucional."""
    model.eval()
    features = []
    grads = []

    # Captura la salida y gradientes de la ultima capa conv
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    # Registrar hooks en la ultima capa convolucional de EfficientNet
    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Remover hooks
    fh.remove()
    bh.remove()

    # Calcular heatmap
    pooled_grads = grads[0].mean(dim=[0, 2, 3])
    feature_map  = features[0][0]

    for i in range(feature_map.shape[0]):
        feature_map[i] *= pooled_grads[i]

    heatmap = feature_map.mean(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Superpone el heatmap sobre la imagen original."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed