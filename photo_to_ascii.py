
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from rembg import remove

SRC = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
COLS = 80
ASPECT = 0.85
DETAIL = 3.2
WEIGHT = 0.5
BUST = 1.0
RAMP = "@$B%8&WM#*oahkbdpqwmZO0QLCJUXYzcvuxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. "
KEEP_FULL_IMAGE = False  # Set to False to remove background and crop to bust

def main():
    img = Image.open(SRC)

    if KEEP_FULL_IMAGE:
        g = np.asarray(ImageOps.autocontrast(img.convert("L"), cutoff=1), dtype=np.int16)
        h, w = g.shape
        
        # local contrast
        blur = np.asarray(Image.fromarray(g.astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(max(2, w // 55))), dtype=np.int16)
        ink = np.clip(150 + (g - blur) * DETAIL + (g - 128) * WEIGHT, 0, 255)
        
        lo, hi = np.percentile(ink, 2), np.percentile(ink, 98)
        ink = np.clip((ink - lo) * 255.0 / max(1, hi - lo), 0, 255)
        
        rows = max(1, int(COLS * (h / w) / ASPECT))
        small = np.asarray(Image.fromarray(ink.astype(np.uint8))
                           .resize((COLS, rows), Image.LANCZOS), dtype=float)
        
        n = len(RAMP) - 1
        lines = []
        for y in range(rows):
            line = "".join(
                RAMP[round(small[y, x] / 255 * n)]
                for x in range(COLS)
            )
            lines.append(line)
    else:
        cut = remove(img)                       # cut the subject out of the background
        rgba = np.asarray(cut)
        alpha = rgba[:, :, 3]

        ys, xs = np.nonzero(alpha > 60)
        x0, x1 = xs.min(), xs.max()
        y0 = ys.min()
        y1 = int(y0 + (ys.max() - y0) * BUST)               # head + torso only
        pad = 8
        box = (max(0, x0 - pad), max(0, y0 - pad),
               min(rgba.shape[1], x1 + pad), min(rgba.shape[0], y1))

        cut = cut.crop(box)
        a = np.asarray(cut)[:, :, 3].astype(float) / 255.0
        g = np.asarray(ImageOps.autocontrast(cut.convert("L"), cutoff=1), dtype=np.int16)
        h, w = g.shape

        # local contrast: pulls folds/edges out of the flat dark shirt
        blur = np.asarray(Image.fromarray(g.astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(max(2, w // 55))), dtype=np.int16)
        ink = np.clip(150 + (g - blur) * DETAIL + (g - 128) * WEIGHT, 0, 255)

        inside = a > 0.5
        lo, hi = np.percentile(ink[inside], 2), np.percentile(ink[inside], 98)
        ink = np.clip((ink - lo) * 255.0 / max(1, hi - lo), 0, 255)

        rows = max(1, int(COLS * (h / w) / ASPECT))
        small = np.asarray(Image.fromarray(ink.astype(np.uint8))
                           .resize((COLS, rows), Image.LANCZOS), dtype=float)
        mask = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                          .resize((COLS, rows), Image.LANCZOS), dtype=float)

        n = len(RAMP) - 1
        lines_light = []
        lines_dark = []
        for y in range(rows):
            line_l = "".join(
                RAMP[round(small[y, x] / 255 * n)] if mask[y, x] > 110 else " "
                for x in range(COLS)
            )
            lines_light.append(line_l.rstrip())
            
            line_d = "".join(
                RAMP[n - round(small[y, x] / 255 * n)] if mask[y, x] > 110 else " "
                for x in range(COLS)
            )
            lines_dark.append(line_d.rstrip())

    parent_dir = Path(__file__).parent
    parent_dir.joinpath("portrait_light.txt").write_text("\n".join(lines_light), encoding="utf-8")
    parent_dir.joinpath("portrait_dark.txt").write_text("\n".join(lines_dark), encoding="utf-8")
    # write default portrait.txt as light
    parent_dir.joinpath("portrait.txt").write_text("\n".join(lines_light), encoding="utf-8")
    print("\n".join(lines_light))
    print(f"\nwrote portrait_light.txt and portrait_dark.txt ({COLS} cols x {rows} rows)")


if __name__ == "__main__":
    main()
