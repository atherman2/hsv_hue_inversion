"""
Usage:
    python highboost.py input.png m k [output.png]

m: size of the mxm box (average) filter
k: high-boost weight (k = 1 -> unsharp masking, k > 1 -> high-boost)
"""

import sys
import numpy as np
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter


def high_boost(image, m, k):
    # image as float to allow negative values during the process
    f = img_as_float(image).astype(np.float64)

    # blurred version using simple average (box) filter mxm
    f_blur = uniform_filter(f, size=m)

    # mask = original - blurred
    mask = f - f_blur

    # g = f + k * mask
    g = f + k * mask

    # clip back to valid range [0, 1] and convert to uint8
    g = np.clip(g, 0, 1)
    g = (g * 255).astype(np.uint8)

    return g


def main():
    if len(sys.argv) < 4:
        print("Usage: python highboost.py input.png m k [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    m = int(sys.argv[2])
    k = float(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) > 4 else "output.png"

    image = io.imread(input_path, as_gray=True)

    result = high_boost(image, m, k)

    io.imsave(output_path, result)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()