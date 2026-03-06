#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片金额和日期修改工具 - 通用版
用户可以自定义参数来调整覆盖位置
"""

import os
from PIL import Image, ImageDraw, ImageFont

# =====================================================
# 在这里修改参数！根据需要调整这些值
# =====================================================

# 输入输出
INPUT_IMAGE = r"C:\Users\ZhanYi\PycharmProjects\AiCustomer\image\bb8477d64fb257eabe5ff2b0b4564fd96ad1f328a6e0f1a513fc7e87c6b39c74.jpg"
OUTPUT_IMAGE = None  # 如果为None，将自动生成

# 新金额和新日期
NEW_AMOUNT = "₹500"
NEW_DATE = "02 Mar 2026"

# =====================================================
# 区域覆盖参数 (0.0-1.0 表示相对于图片尺寸的比例)
# =====================================================

# 顶部金额区域 (蓝色卡片区域)
TOP_AMOUNT = {
    'x1': 0.10, 'y1': 0.04,   # 左上角
    'x2': 0.90, 'y2': 0.38,   # 右下角 (增大区域)
    'font_size_amount': 60,
    'font_size_words': 24,
    'color': (255, 255, 255),  # 白色
    'fill_color': (64, 156, 255)  # Paytm蓝色
}

# 中间金额区域 (₹100显示位置)
MID_AMOUNT = {
    'x1': 0.15, 'y1': 0.17,   # 增大区域确保完全覆盖
    'x2': 0.85, 'y2': 0.35,
    'font_size': 30,
    'color': (0, 128, 0),  # 绿色
    'fill_color': (64, 156, 255)  # 使用蓝色保持一致
}

# 底部日期时间区域
BOTTOM_DATE = {
    'x1': 0.08, 'y1': 0.83,
    'x2': 0.92, 'y2': 0.94,
    'font_size': 26,
    'color': (80, 80, 80),  # 灰色
    'fill_color': (255, 255, 255)  # 白色
}

# =====================================================

class ImageTextEditor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size
    
    def cover_rect(self, x1, y1, x2, y2, fill_color):
        """用坐标覆盖矩形区域"""
        draw = ImageDraw.Draw(self.image)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)
    
    def add_text_centered(self, x, y, text, font_size, color):
        """居中添加文字"""
        draw = ImageDraw.Draw(self.image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x - text_width // 2
        text_y = y - text_height // 2
        
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    def add_text(self, x, y, text, font_size, color):
        """添加文字"""
        draw = ImageDraw.Draw(self.image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((x, y), text, fill=color, font=font)
    
    def number_to_words(self, n):
        """将数字转换为英文单词"""
        try:
            n = int(n)
        except:
            return "Five Hundred"
        
        mapping = {
            100: "One Hundred", 200: "Two Hundred", 300: "Three Hundred",
            400: "Four Hundred", 500: "Five Hundred", 600: "Six Hundred",
            700: "Seven Hundred", 800: "Eight Hundred", 900: "Nine Hundred",
            1000: "One Thousand", 2000: "Two Thousand", 5000: "Five Thousand",
            10000: "Ten Thousand"
        }
        return mapping.get(n, f"{n}")
    
    def process(self, new_amount, new_date, output_path=None):
        """处理图片"""
        
        w, h = self.width, self.height
        
        # 1. 覆盖顶部金额区域
        print("正在处理金额...")
        cfg = TOP_AMOUNT
        x1, y1 = int(w * cfg['x1']), int(h * cfg['y1'])
        x2, y2 = int(w * cfg['x2']), int(h * cfg['y2'])
        self.cover_rect(x1, y1, x2, y2, cfg['fill_color'])
        
        # 添加新金额
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.add_text_centered(cx, cy - 25, new_amount, cfg['font_size_amount'], cfg['color'])
        
        amount_words = self.number_to_words(new_amount.replace('₹', '').strip())
        self.add_text_centered(cx, cy + 25, f"Rupees {amount_words} Only", cfg['font_size_words'], cfg['color'])
        
        # 2. 覆盖中间金额区域
        cfg = MID_AMOUNT
        x1, y1 = int(w * cfg['x1']), int(h * cfg['y1'])
        x2, y2 = int(w * cfg['x2']), int(h * cfg['y2'])
        self.cover_rect(x1, y1, x2, y2, cfg['fill_color'])
        
        # 添加勾选图标和金额
        self.add_text(x1 + 30, y1 + 5, "✓", cfg['font_size'], cfg['color'])
        self.add_text(x1 + 80, y1 + 5, new_amount, cfg['font_size'], cfg['color'])
        
        # 添加金额文字
        mid_cx = (x1 + x2) // 2
        amount_words = self.number_to_words(new_amount.replace('₹', '').strip())
        self.add_text_centered(mid_cx, y1 + 40, f"Rupees {amount_words} Only", cfg['font_size'] - 4, cfg['color'])
        
        # 3. 覆盖底部日期时间
        print("正在处理日期...")
        cfg = BOTTOM_DATE
        x1, y1 = int(w * cfg['x1']), int(h * cfg['y1'])
        x2, y2 = int(w * cfg['x2']), int(h * cfg['y2'])
        self.cover_rect(x1, y1, x2, y2, cfg['fill_color'])
        
        # 添加新日期时间
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.add_text_centered(cx, cy, f"08:01 PM, {new_date}", cfg['font_size'], cfg['color'])
        
        # 4. 保存结果
        if output_path is None:
            base, ext = os.path.splitext(self.image_path)
            output_path = f"{base}_edited{ext}"
        
        self.image.save(output_path)
        print(f"图片已保存至: {output_path}")
        
        return output_path


def main():
    print("=" * 60)
    print("图片金额和日期修改工具 - 通用版")
    print("=" * 60)
    print(f"输入图片: {INPUT_IMAGE}")
    print(f"新金额: {NEW_AMOUNT}")
    print(f"新日期: {NEW_DATE}")
    print("=" * 60)
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"错误: 图片文件不存在: {INPUT_IMAGE}")
        return
    
    editor = ImageTextEditor(INPUT_IMAGE)
    output_path = editor.process(NEW_AMOUNT, NEW_DATE, OUTPUT_IMAGE)
    
    print(f"\n完成！输出文件: {output_path}")


if __name__ == "__main__":
    main()
