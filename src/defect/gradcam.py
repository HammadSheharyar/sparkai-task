import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, layer, x):
    """x: 1xCxHxW. Returns (cam HxW in [0,1], defect probability)."""
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.eval()
        with torch.enable_grad():
            logit = model(x).squeeze()
            model.zero_grad()
            logit.backward()
    finally:
        h.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()
    return cam, float(torch.sigmoid(logit).item())


def composite(img_rgb: np.ndarray, cam: np.ndarray, title: str) -> np.ndarray:
    """side-by-side [original | grad-cam overlay] as BGR uint8 for cv2.imwrite."""
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_rgb + 0.45 * heat).astype(np.uint8)
    both = np.concatenate([img_rgb, overlay], axis=1)
    bar = np.full((28, both.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.concatenate([bar, both], axis=0), cv2.COLOR_RGB2BGR)
