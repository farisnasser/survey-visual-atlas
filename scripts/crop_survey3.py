"""Crop every figure of survey v3 from its PDF into public/survey3/figN.webp (for :::survey blocks in guide/), and copy
the PDF itself. Each figure = the drawings above its 'Fig. N:' caption in the caption's column (or full width)."""
import fitz, os, re, shutil, io
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "..", "survey-v3", "Optimization_Methods_Survey_with_Figures.pdf")
OUT = os.path.join(ROOT, "public", "survey3"); os.makedirs(OUT, exist_ok=True)
shutil.copy(SRC, os.path.join(OUT, "Optimization_Methods_Survey_with_Figures.pdf"))
d = fitz.open(SRC)
done = set()
for p in d:
    for b in p.get_text("dict")["blocks"]:
        txt = "".join(s["text"] for l in b.get("lines", []) for s in l["spans"])
        m = re.match(r"Fig\. (\d+):", txt)
        if not m or m.group(1) in done:
            continue
        n = m.group(1); cap = fitz.Rect(b["bbox"])
        x0, x1 = cap.x0 - 4, cap.x1 + 4
        rs = [dr["rect"] for dr in p.get_drawings() if dr["rect"].y1 <= cap.y0 + 1 and dr["rect"].y0 >= cap.y0 - 340
              and dr["rect"].x0 >= x0 - 2 and dr["rect"].x1 <= x1 + 2]
        ims = [fitz.Rect(i["bbox"]) for i in p.get_image_info() if i["bbox"][3] <= cap.y0 + 1 and i["bbox"][1] >= cap.y0 - 340]
        rs += ims
        # never climb past another caption, table or text block that sits above this figure in the same column
        lim = 36
        for ob in p.get_text("dict")["blocks"]:
            r = fitz.Rect(ob["bbox"]); t2 = "".join(s["text"] for l in ob.get("lines", []) for s in l["spans"])
            if r.y1 < cap.y0 - 2 and r.x1 > x0 + 5 and r.x0 < x1 - 5 and re.match(r"(Fig\. \d+:|TABLE [IVX]+:)", t2):
                lim = max(lim, r.y1 + 2)
        rs = [r for r in rs if r.y0 >= lim]
        if not rs:
            print("no drawings for Fig.", n); continue
        top = min(r.y0 for r in rs) - 10
        clip = fitz.Rect(x0, max(top, 36), x1, cap.y0 - 1)
        pix = p.get_pixmap(dpi=230, clip=clip)
        im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        im.save(os.path.join(OUT, f"fig{n}.webp"), "WEBP", quality=88)
        done.add(n); print(f"Fig. {n}: page {p.number + 1}, {im.size}")
print(sorted(done, key=int))
