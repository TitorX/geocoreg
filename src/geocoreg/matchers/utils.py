import numpy as np
import cv2

def apply_clahe(img):
    """Apply CLAHE normalization to a grayscale uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def robust_8bit_normalize(arr):
    """2-98 percentile stretch to uint8 while preserving channel count.

    Handles any input dtype (uint8, uint16, float32, etc.) by stretching
    the valid (>0) pixel range to [0, 255]. For multi-channel inputs,
    stretching is applied per-channel and the original number of channels
    is preserved.
    """
    if arr is None or arr.size == 0:
        return np.zeros(arr.shape[:2] if arr is not None else (1, 1), dtype=np.uint8)

    arr_f = arr.astype(np.float64)

    if arr_f.ndim == 2:
        valid = arr_f > 0
        if not np.any(valid):
            return np.zeros(arr_f.shape, dtype=np.uint8)

        p2, p98 = np.percentile(arr_f[valid], (2, 98))
        if p98 <= p2:
            p98 = p2 + 1e-5

        stretched = np.clip((arr_f - p2) / (p98 - p2) * 255.0, 0, 255)
        out = stretched.astype(np.uint8)
        out[~valid] = 0
        return out

    if arr_f.ndim == 3:
        out = np.zeros(arr_f.shape, dtype=np.uint8)
        for ch in range(arr_f.shape[2]):
            plane = arr_f[:, :, ch]
            valid = plane > 0
            if not np.any(valid):
                continue

            p2, p98 = np.percentile(plane[valid], (2, 98))
            if p98 <= p2:
                p98 = p2 + 1e-5

            stretched = np.clip((plane - p2) / (p98 - p2) * 255.0, 0, 255)
            plane_u8 = stretched.astype(np.uint8)
            plane_u8[~valid] = 0
            out[:, :, ch] = plane_u8
        return out

    raise ValueError(f"Unsupported array shape for normalization: {arr.shape}")
