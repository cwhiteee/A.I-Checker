
import io
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import cv2
from PIL import Image, ExifTags

@dataclass
class FeatureScores:
    exif_ai_likely: float
    symmetry_ai_likely: float
    texture_ai_likely: float
    background_ai_likely: float
    spectrum_ai_likely: float
    color_ai_likely: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "exif_ai_likely": self.exif_ai_likely,
            "symmetry_ai_likely": self.symmetry_ai_likely,
            "texture_ai_likely": self.texture_ai_likely,
            "background_ai_likely": self.background_ai_likely,
            "spectrum_ai_likely": self.spectrum_ai_likely,
            "color_ai_likely": self.color_ai_likely,
        }

CAMERA_EXIF_KEYS = {"Make","Model","LensModel","LensMake","DateTimeOriginal","FNumber","ExposureTime","ISOSpeedRatings","FocalLength"}

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()

def _center_crop(img: np.ndarray, frac: float=0.6) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h*frac), int(w*frac)
    y1 = (h - ch)//2
    x1 = (w - cw)//2
    return img[y1:y1+ch, x1:x1+cw]

def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-9)
    nz = p[p>0]
    return float(-(nz * np.log2(nz)).sum())

def exif_score(pil_img: Image.Image) -> float:
    # 1.0 means "looks AI" because missing or generic EXIF
    score = 1.0
    try:
        exif = pil_img.getexif()
        if exif is None or len(exif) == 0:
            return 0.95
        # Map tags
        tags = {}
        for k,v in exif.items():
            tag = ExifTags.TAGS.get(k, str(k))
            tags[tag] = v
        # Count meaningful camera fields
        meaningful = sum(1 for k in CAMERA_EXIF_KEYS if k in tags)
        # If we have many real camera fields, less likely AI
        if meaningful >= 3:
            score = 0.2
        elif meaningful == 2:
            score = 0.4
        elif meaningful == 1:
            score = 0.6
        else:
            score = 0.85
    except Exception:
        score = 0.95
    return float(np.clip(score, 0.0, 1.0))

def symmetry_score(bgr: np.ndarray) -> float:
    # Compare left/right halves in the central crop
    c = _center_crop(bgr, 0.6)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid = w//2
    left = gray[:, :mid]
    right = gray[:, w-mid:][:, ::-1]  # mirror
    # Normalized mean absolute difference
    mad = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0
    # Very small difference => suspicious (AI loves symmetry)
    # Map MAD in [0,0.15] -> AI score 1->0; else 0
    score = 1.0 - np.clip(mad/0.15, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def texture_scores(bgr: np.ndarray) -> Tuple[float, float]:
    # Returns (texture_ai_likely, background_ai_likely)
    img = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)
    c = _center_crop(img, 0.6)
    # Laplacian variance: micro-texture
    def lap_var(x):
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    face_tex = lap_var(c)
    edges = img.copy()
    hh, ww = img.shape[:2]
    m = 32
    edges[:(hh - c.shape[0])//2 + c.shape[0], (ww - c.shape[1])//2:(ww + c.shape[1])//2] = 0
    # take outer bands as "background-like"
    mask = np.ones((hh,ww), np.uint8)*255
    y1 = (hh - c.shape[0])//2
    x1 = (ww - c.shape[1])//2
    mask[y1:y1+c.shape[0], x1:x1+c.shape[1]] = 0
    bg = cv2.bitwise_and(img, img, mask=mask)
    # Compute texture only where mask>0
    gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    # Zero pixels ignored by selecting >0
    vals = gray_bg[mask>0]
    if vals.size == 0:
        bg_tex = 0.0
    else:
        # Laplacian variance on background region
        lap_bg = cv2.Laplacian(gray_bg, cv2.CV_64F)
        bg_tex = float(lap_bg[mask>0].var())

    # Heuristic mapping:
    # Very smooth skin (low face_tex) -> AI likely
    # Very smooth background compared to face -> AI likely
    # Normalize by empirical constants
    f_norm = min(face_tex / 200.0, 3.0)  # typical camera faces 0.8-1.5
    b_norm = min(bg_tex / 150.0, 3.0)

    texture_ai = float(np.clip(1.2 - f_norm, 0.0, 1.0))  # low var -> closer to 1
    background_ai = float(np.clip(0.9 - b_norm, 0.0, 1.0))  # very smooth bg -> closer to 1
    return texture_ai, background_ai

def spectrum_score(bgr: np.ndarray) -> float:
    # Frequency-domain analysis for periodic peaks / checkerboardiness
    gray = cv2.cvtColor(cv2.resize(bgr, (512,512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    # Radial profile
    h, w = mag.shape
    cy, cx = h//2, w//2
    y, x = np.indices((h,w))
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r = r.astype(np.int32)
    maxr = min(cy, cx)
    radial = np.bincount(r.ravel(), mag.ravel()) / (np.bincount(r.ravel()) + 1e-9)
    radial = radial[:maxr]

    # Compare to an ideal 1/f falloff (log-linear). Compute deviations (peaks)
    # Fit simple line in log-log
    eps = 1e-6
    xs = np.arange(1, len(radial))
    ys = radial[1:]
    logx = np.log(xs + eps)
    logy = np.log(ys + eps)
    A = np.vstack([logx, np.ones_like(logx)]).T
    m, c = np.linalg.lstsq(A, logy, rcond=None)[0]
    # Residuals
    pred = m*logx + c
    resid = logy - pred
    # Peakiness: std of positive residuals
    peakiness = float(np.std(resid[resid>0]))

    # Map peakiness to AI likelihood
    # More peaks -> more likely synthesized/upscaled
    score = np.clip((peakiness - 0.15) / 0.2, 0.0, 1.0)
    return float(score)

def color_score(bgr: np.ndarray) -> float:
    # Low hue entropy and overly tight saturation -> suspicious
    hsv = cv2.cvtColor(cv2.resize(bgr, (512,512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    h_hist = np.histogram(h, bins=36, range=(0,180))[0]
    s_hist = np.histogram(s, bins=32, range=(0,256))[0]

    h_entropy = _entropy(h_hist)
    s_entropy = _entropy(s_hist)

    # Map: low entropy (esp. hue) -> AI
    # Typical real photos: hue_entropy ~ 4.5-5.2 (36 bins)
    # We'll flag <= 4.0 as suspicious
    h_ai = np.clip((4.2 - h_entropy)/1.2, 0.0, 1.0)
    # Very tight saturation spread can be suspicious too
    s_ai = np.clip((4.2 - s_entropy)/1.2, 0.0, 1.0)
    return float(np.clip(0.6*h_ai + 0.4*s_ai, 0.0, 1.0))

def detect(pil_img: Image.Image) -> Tuple[float, FeatureScores, Dict[str, Any]]:
    """
    Returns:
      - ai_probability (0..1)
      - per-feature scores (0..1 each, higher = more AI-like)
      - debug dict with raw measures
    """
    bgr = _pil_to_bgr(pil_img)

    exif = exif_score(pil_img)
    sym = symmetry_score(bgr)
    tex, bg = texture_scores(bgr)
    spec = spectrum_score(bgr)
    col = color_score(bgr)

    features = FeatureScores(
        exif_ai_likely=exif,
        symmetry_ai_likely=sym,
        texture_ai_likely=tex,
        background_ai_likely=bg,
        spectrum_ai_likely=spec,
        color_ai_likely=col
    )

    # Weighted fusion (heuristic). Tuned to be conservative for portraits.
    weights = {
        "exif_ai_likely": 0.22,
        "symmetry_ai_likely": 0.25,
        "texture_ai_likely": 0.18,
        "background_ai_likely": 0.07,
        "spectrum_ai_likely": 0.08,
        "color_ai_likely": 0.20,
    }
    total = 0.0
    for k,v in features.as_dict().items():
        total += weights[k]*v

    # Clamp to [0,1]
    ai_prob = float(np.clip(total, 0.0, 1.0))

    debug = {
        "weights": weights,
        "feature_scores": features.as_dict(),
    }
    return ai_prob, features, debug
