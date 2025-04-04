import argparse
from utils.process_image import process_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glitch an image using databending techniques.")
    parser.add_argument("--input", "-i", required=True, help="Path to input image (PNG/TIF)")
    parser.add_argument("--output", "-o", required=True, help="Path to save glitched output image")
    parser.add_argument("--mode", "-m", required=True, help="Glitch mode to apply (e.g., bass_treble, pitch_shift)")

    args = parser.parse_args()

    process_image(args.input, args.output, args.mode)
