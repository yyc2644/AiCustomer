import re
import argparse
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os
import statistics


def parse_region(s: str):
    # format: x,y,w,h
    parts = s.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('Region must be x,y,w,h')
    return tuple(int(p) for p in parts)


def find_text_boxes(img):
    # returns list of dicts with text and bbox
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        boxes.append({'text': text, 'bbox': (x, y, w, h)})
    return boxes


def find_amount_and_time_boxes(img):
    boxes = find_text_boxes(img)
    amount_box = None
    time_box = None
    amount_re = re.compile(r'^[¥$€£]?[0-9][0-9,\.]*$')
    time_re = re.compile(r'^[0-2]?\d:[0-5]\d$')
    # search for best matches
    for b in boxes:
        t = b['text']
        if time_box is None and time_re.match(t):
            time_box = b
            continue
        if amount_box is None and amount_re.match(t):
            amount_box = b
    return amount_box, time_box


def estimate_colors(img, bbox):
    # bbox: (x,y,w,h)
    x, y, w, h = bbox
    pad = 4
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img.width, x + w)
    y1 = min(img.height, y + h)
    region = img.crop((x0, y0, x1, y1)).convert('RGBA')
    pixels = list(region.getdata())
    # find background color by sampling border pixels around the bbox
    bg_candidates = []
    # sample a thin border around the bbox from original image
    bx0 = max(0, x - pad)
    by0 = max(0, y - pad)
    bx1 = min(img.width, x + w + pad)
    by1 = min(img.height, y + h + pad)
    border = img.crop((bx0, by0, bx1, by1)).convert('RGBA')
    border_pixels = list(border.getdata())
    # choose background as the most common color in border
    try:
        bg = max(set(border_pixels), key=border_pixels.count)
    except Exception:
        bg = (255, 255, 255, 255)

    # choose text color by taking darkest pixels in region
    def luminance(px):
        r, g, b, a = px
        return 0.299 * r + 0.587 * g + 0.114 * b

    pixels_sorted = sorted(pixels, key=luminance)
    # take mean of darkest 10% pixels
    take = max(1, len(pixels_sorted) // 10)
    darkest = pixels_sorted[:take]
    rs = [p[0] for p in darkest]
    gs = [p[1] for p in darkest]
    bs = [p[2] for p in darkest]
    fg = (int(statistics.mean(rs)), int(statistics.mean(gs)), int(statistics.mean(bs)))
    return fg, bg[:3]


def cover_and_draw(img, bbox, new_text, font_path=None):
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    fg, bg = estimate_colors(img, bbox)
    # fill rectangle with background color
    draw.rectangle([x - 1, y - 1, x + w + 1, y + h + 1], fill=bg)
    # choose font size roughly matching bbox height
    font_size = max(10, int(h * 0.9))
    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        # try common Windows font
        try:
            font = ImageFont.truetype('arial.ttf', font_size)
        except Exception:
            font = ImageFont.load_default()

    # compute text size and adjust position to keep same bbox alignment
    text_w, text_h = draw.textsize(new_text, font=font)
    # center vertically within bbox, align left with original x
    tx = x
    ty = y + (h - text_h) // 2
    draw.text((tx, ty), new_text, fill=fg, font=font)


def main():
    parser = argparse.ArgumentParser(description='Replace amount and time on an image')
    parser.add_argument('--image', '-i', default=r'image/bb8477d64fb257eabe5ff2b0b4564fd96ad1f328a6e0f1a513fc7e87c6b39c74.jpg')
    parser.add_argument('--amount', '-a', required=True, help='New amount text, e.g. $12.34')
    parser.add_argument('--time', '-t', required=True, help='New time text, e.g. 12:34')
    parser.add_argument('--amount-region', type=parse_region, help='Manual region for amount x,y,w,h')
    parser.add_argument('--time-region', type=parse_region, help='Manual region for time x,y,w,h')
    parser.add_argument('--font', help='Path to .ttf font to use (optional)')
    parser.add_argument('--out', '-o', default='output.png')
    args = parser.parse_args()

    img = Image.open(args.image).convert('RGBA')

    # locate boxes
    if args.amount_region or args.time_region:
        amount_box = {'bbox': args.amount_region} if args.amount_region else None
        time_box = {'bbox': args.time_region} if args.time_region else None
    else:
        a_box, t_box = find_amount_and_time_boxes(img)
        amount_box = a_box
        time_box = t_box

    if amount_box is None:
        print('Warning: amount box not found. Use --amount-region to supply x,y,w,h')
    else:
        cover_and_draw(img, amount_box['bbox'], args.amount, font_path=args.font)

    if time_box is None:
        print('Warning: time box not found. Use --time-region to supply x,y,w,h')
    else:
        cover_and_draw(img, time_box['bbox'], args.time, font_path=args.font)

    img.save(args.out)
    print(f'Saved output to {args.out}')


if __name__ == '__main__':
    main()
