import numpy as np
import audioop
from scipy.fft import fft, ifft

# -------------------------------
#    U-Law Encoding & Decoding
# -------------------------------

def ulaw_encode(data: bytes) -> bytes:
    """Encode raw 8-bit data to U-Law."""
    return audioop.lin2ulaw(data, 1)

def ulaw_decode(data: bytes) -> bytes:
    """Decode U-Law data back to linear."""
    return audioop.ulaw2lin(data, 1)

# -------------------------------
#        Header Protection
# -------------------------------

def apply_glitch_with_header_protection(data: bytes, glitch_func, header_len=512) -> bytearray:
    """
    Splits off the image header, applies glitch_func to the body, and reattaches the header.
    """
    header = data[:header_len]
    body = data[header_len:]

    glitched_body = glitch_func(body)
    if len(glitched_body) != len(body):
        # Resize to match original body length for safe reconstruction
        if len(glitched_body) > len(body):
            glitched_body = glitched_body[:len(body)]
        else:
            pad_len = len(body) - len(glitched_body)
            glitched_body = glitched_body + bytes([0] * pad_len)

    return bytearray(header + glitched_body)

# -------------------------------
#   Bass & Treble Glitch Effect
# -------------------------------

def glitch_bass_treble(data: bytes, width: int, block_height: int = 16) -> bytearray:
    """
    Applies bass & treble boosting to random vertical blocks of the image,
    treating each color channel independently. This introduces more glitchy,
    distorted effects than the global version.
    """
    import numpy as np
    from scipy.fft import fft, ifft

    arr = np.frombuffer(data, dtype=np.uint8)
    bytes_per_pixel = 3
    row_stride = width * bytes_per_pixel
    total_rows = len(arr) // row_stride

    # Trim extra bytes
    arr = arr[:total_rows * row_stride]
    img_3d = arr.reshape((total_rows, width, 3))  # (H, W, C)

    blocks = [
        img_3d[i:i + block_height]
        for i in range(0, total_rows, block_height)
    ]

    glitched_blocks = []
    for block in blocks:
        block = block.copy()
        for c in range(3):  # For each color channel
            channel = block[:, :, c].astype(np.float32)
            norm = (channel - 127.5) / 127.5

            # Flatten channel for FFT, then reshape back
            flat = norm.flatten()
            freq = fft(flat)

            # Boost bass & treble with random gains
            n = len(freq)
            bass_end = int(n * 0.05)
            treble_start = int(n * 0.95)

            bass_gain = np.random.uniform(1.5, 3.5)
            treble_gain = np.random.uniform(1.5, 3.5)

            freq[:bass_end] *= bass_gain
            freq[treble_start:] *= treble_gain

            altered = np.real(ifft(freq)).reshape(channel.shape)
            out = np.clip((altered * 127.5) + 127.5, 0, 255).astype(np.uint8)
            block[:, :, c] = out

        glitched_blocks.append(block)

    glitched_img = np.vstack(glitched_blocks)
    return bytearray(glitched_img.flatten())

# -------------------------------
#    Change Pitch Glitch Effect
# -------------------------------

def glitch_pitch_shift(data: bytes, factor=1.2) -> bytearray:
    """
    Simulates pitch change by resampling the byte stream.
    Factor > 1 = lower pitch (stretch); < 1 = higher pitch (compress).
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)

    indices = np.linspace(0, n - 1, int(n * factor)).astype(np.int32)
    stretched = arr[indices % n]

    return bytearray(stretched[:n])  # trim to original length

# -------------------------------
#    Change Speed Glitch Effect
# -------------------------------

def glitch_change_speed(data: bytes, speed_factor=1.5, target_length=None) -> bytearray:
    # Step 1: Encode to u-law (like importing image as sound in Audacity)
    encoded = ulaw_encode(data)

    # Step 2: Resample the encoded data
    arr = np.frombuffer(encoded, dtype=np.uint8)
    new_len = int(len(arr) / speed_factor)
    indices = np.linspace(0, len(arr) - 1, new_len).astype(np.int32)
    resampled = arr[indices % len(arr)]

    # Step 3: Pad or trim to match original length
    if target_length:
        original_len = target_length
    else:
        original_len = len(data)

    if len(resampled) > original_len:
        resampled = resampled[:original_len]
    else:
        pad_len = original_len - len(resampled)
        resampled = np.pad(resampled, (0, pad_len), mode='wrap')

    # Step 4: Decode back to linear space
    final = ulaw_decode(resampled.tobytes())
    return bytearray(final[:original_len])

# -------------------------------
#    Change Tempo Glitch Effect
# -------------------------------

def glitch_change_tempo(data: bytes, width: int, block_height: int = 16, stretch_factor: float = 2.0) -> bytearray:
    """
    Simulates Audacity's Change Tempo effect visually by repeating or compressing
    vertical blocks of the image data (glitch echo bands).

    Parameters:
    - data: Raw bytes of the uncompressed RGB image.
    - width: Image width in pixels.
    - block_height: Height of each block to repeat/compress.
    - stretch_factor: >1.0 repeats blocks, <1.0 compresses.
    """

    arr = np.frombuffer(data, dtype=np.uint8)
    bytes_per_pixel = 3
    row_stride = width * bytes_per_pixel

    # Reshape the image as (H, row_stride)
    total_rows = len(arr) // row_stride
    arr = arr[:total_rows * row_stride]  # trim if needed
    img_2d = arr.reshape((total_rows, row_stride))

    # Divide into vertical blocks
    blocks = [img_2d[i:i + block_height] for i in range(0, total_rows, block_height)]

    # Repeat or compress blocks
    new_blocks = []
    for block in blocks:
        if stretch_factor > 1:
            repeats = int(stretch_factor)
            for _ in range(repeats):
                new_blocks.append(block)
        else:
            # Probabilistic inclusion for compression
            if np.random.rand() < stretch_factor:
                new_blocks.append(block)

    # Recombine
    glitched_img = np.vstack(new_blocks)
    glitched_img = glitched_img[:total_rows]  # clip to original height

    return bytearray(glitched_img.flatten())