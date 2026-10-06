from PIL import Image, ImageDraw

img = Image.open("data/train/images/aa985f01-d478-3fe4-98f2-6327ed7bdabb.jpg")
W, H = img.size
draw = ImageDraw.Draw(img)

with open("data/train/labels/aa985f01-d478-3fe4-98f2-6327ed7bdabb.txt", "r", encoding="utf-8") as f:
    for line in f.read().splitlines():
        # YOLO label line: class, then box center x, center y, width, height (all as fractions 0..1 of the image size)
        yolo_class, cx, cy, bw, bh = line.split()
        yolo_class, cx, cy, bw, bh = float(yolo_class), float(cx), float(cy), float(bw), float(bh)

        x1 = (cx - bw / 2) * W
        x2 = (cx + bw / 2) * W
        y1 = (cy - bh / 2) * H
        y2 = (cy + bh / 2)  * H

        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)


img.save("splits/draw_check.jpg")
print(img.size)



