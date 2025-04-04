import os
from PIL import Image
import numpy as np
from utils.glitch_modes import GLITCH_MODES
from utils.image_glitch import apply_glitch_with_header_protection

# -------------------------------
#        Image Processor
# -------------------------------

def process_image(img_path, output_path, glitch_mode, header_len=512):
    # Step 1: Convert to uncompressed TIF
    temp_input_path = img_path.rsplit(".", 1)[0] + "_temp_input.tif"
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    img.save(temp_input_path, format="TIFF", compression="none")
    print(f"Converted {img_path} to uncompressed TIF: {temp_input_path}")

    glitch_entry = GLITCH_MODES.get(glitch_mode)
    if glitch_entry is None:
        raise ValueError(f"Unknown glitch mode: {glitch_mode}")
    glitch_func = glitch_entry["func"]
    per_channel = glitch_entry.get("per_channel", False)
    needs_width = glitch_entry.get("needs_width", False)

    if per_channel:
        r, g, b = img.split()

        def glitch_channel(channel_bytes):
            return apply_glitch_with_header_protection(
                channel_bytes,
                lambda d: glitch_func(d, width=width) if needs_width else glitch_func(d),
                header_len=header_len,
            )

        glitched_r = glitch_channel(r.tobytes())
        glitched_g = glitch_channel(g.tobytes())
        glitched_b = glitch_channel(b.tobytes())

        r_img = Image.frombytes("L", img.size, glitched_r)
        g_img = Image.frombytes("L", img.size, glitched_g)
        b_img = Image.frombytes("L", img.size, glitched_b)

        glitched_img = Image.merge("RGB", (r_img, g_img, b_img))
        glitched_img.save(output_path)
        print(f"✅ Final PNG output saved to {output_path}")

    else:
        with open(temp_input_path, "rb") as f:
            data = f.read()

        glitched_data = apply_glitch_with_header_protection(
            data,
            lambda d: glitch_func(d, width=width) if needs_width else glitch_func(d),
            header_len=header_len,
        )

        pixel_data = glitched_data[header_len:]
        expected_length = width * height * 3

        if len(pixel_data) < expected_length:
            pad_len = expected_length - len(pixel_data)
            print(f"Warning: Glitched pixel data too short. Padding with {pad_len} zeros.")
            pixel_data += bytes([0] * pad_len)
        elif len(pixel_data) > expected_length:
            print(f"Warning: Glitched pixel data too long. Trimming excess.")
            pixel_data = pixel_data[:expected_length]

        pixel_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        glitched_img = Image.fromarray(pixel_array, mode="RGB")
        glitched_img.save(output_path)
        print(f"✅ Final PNG output saved to {output_path}")

    os.remove(temp_input_path)
