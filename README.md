
# AI Image Detector — Heuristic v2

This is a self-contained, **no-internet** Streamlit app that flags likely **AI-generated** images using a fusion of handcrafted signals:

- **EXIF check** (missing or generic EXIF looks suspicious)
- **Facial symmetry** (AI portraits are often overly symmetric)
- **Skin micro-texture** (overly smooth = suspicious)
- **Background smoothness** (uniform, low-detail backgrounds are common in AI portraits)
- **Frequency spectrum peakiness** (periodic artifacts from synthesis/upscaling)
- **Color distribution entropy** (unnaturally tight hue/saturation ranges)

It returns a probability-like score and a clear verdict. This is **not** forensic proof—treat it as a triage tool.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Quick CLI test
You can also score a single image without Streamlit:
```bash
python quick_test.py /path/to/image.jpg
```

## Notes
- This version avoids external downloads and runs fully offline.
- It does not train any ML model; instead, it combines robust, explainable image statistics.
- For production, combine with a server-side model specialized in GAN detection for higher accuracy.
