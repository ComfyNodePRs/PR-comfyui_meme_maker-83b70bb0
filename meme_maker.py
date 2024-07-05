import os
import sys
import torch
import textwrap
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageFont, ImageColor
import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
FONTS_DIR = folder_paths.get_input_directory()

class MemeMaker:
    """
    A node to add meme-style text to an image
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meme_text": ("STRING", {
                    "multiline": True,
                    "default": "Your Meme Text Here"
                }),
                "font": ([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')], ),
                "max_font_size": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 9999,
                    "step": 1
                }),
                "font_color": ("STRING", {
                    "multiline": False,
                    "default": "#FFFFFF"
                }),
                "outline_color": ("STRING", {
                    "multiline": False,
                    "default": "#000000"
                }),
                "outline_width": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
                "horizontal_align": (["left", "center", "right"], {
                    "default": "center"
                }),
                "vertical_align": (["top", "center", "bottom"], {
                    "default": "bottom"
                }),
                "padding_left": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "padding_right": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "padding_top": ("INT", {  # New input for padding_top
                    "default": 20,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "padding_bottom": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "meme_height": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 9999,
                    "step": 1
                }),
            },
            "optional": {
                "image_input": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "Image Processing"

    def execute(self, meme_text, font, max_font_size, font_color, outline_color, outline_width, horizontal_align, vertical_align, padding_left, padding_right, padding_top, padding_bottom, meme_height, image_input=None):
        meme_text = meme_text.upper()
        font_path = os.path.join(FONTS_DIR, font)
        font = ImageFont.truetype(font_path, max_font_size)

        if image_input is not None:
            image_input = T.ToPILImage()(image_input.permute([0, 3, 1, 2])[0]).convert('RGBA')
            width = image_input.width
            height = image_input.height
            image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
        else:
            raise ValueError("image_input is required")

        draw = ImageDraw.Draw(image)
        max_width = width - padding_left - padding_right
        max_height = meme_height

        def get_text_size(text, font):
            lines = textwrap.wrap(text, width=max_width // (font_size // 2))
            return draw.multiline_textbbox((0, 0), "\n".join(lines), font=font)

        font_size = max_font_size
        bbox = get_text_size(meme_text, font)

        while (bbox[2] - bbox[0] > max_width or bbox[3] - bbox[1] > max_height) and font_size > 1:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            bbox = get_text_size(meme_text, font)

        lines = textwrap.wrap(meme_text, width=max_width // (font_size // 2))

        total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
        if vertical_align == "top":
            text_y = padding_top  # Use padding_top here
        elif vertical_align == "center":
            text_y = (height - total_text_height) // 2
        elif vertical_align == "bottom":
            text_y = height - total_text_height - padding_bottom

        for line in lines:
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            if horizontal_align == "left":
                text_x = padding_left
            elif horizontal_align == "center":
                text_x = (width - line_width) // 2
            elif horizontal_align == "right":
                text_x = width - line_width - padding_right

            # Draw outline
            for adj in range(-outline_width, outline_width + 1):
                for adj2 in range(-outline_width, outline_width + 1):
                    if adj != 0 or adj2 != 0:
                        draw.text((text_x + adj, text_y + adj2), line, font=font, fill=outline_color)

            # Draw text
            draw.text((text_x, text_y), line, font=font, fill=font_color)
            text_y += draw.textbbox((0, 0), line, font=font)[3]

        if image_input is not None:
            image = Image.alpha_composite(image_input, image)

        image_tensor_out = T.ToTensor()(image).unsqueeze(0).permute([0, 2, 3, 1])
        mask = image_tensor_out[:, :, :, 3] if image_tensor_out.shape[3] == 4 else torch.ones_like(image_tensor_out[:, :, :, 0])

        return (image_tensor_out[:, :, :, :3], mask,)

NODE_CLASS_MAPPINGS = {
    "MemeMaker": MemeMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemeMaker": "Meme Maker - fonts autoread in inputs folder"
}
