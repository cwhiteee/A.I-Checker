
import sys
from PIL import Image
from detector import detect

def main(path):
    img = Image.open(path)
    p, features, debug = detect(img)
    print(f"AI likelihood: {p*100:.1f}%")
    for k,v in features.as_dict().items():
        print(f"  {k:24s} {v:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py /path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
