from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_binarize_image(image_path, threshold=128):
    """
    Reads an image, converts it to grayscale, and binarizes it.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    return (img_array > threshold).astype(int)

def apply_fixed_convolution(image, kernel):
    """
    Applies a fixed 2D convolution (no padding, stride=1).
    """
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out

if __name__ == "__main__":
    image_path = "../images/2_0.png"  # adjust as needed
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        exit()

    binary_img = read_and_binarize_image(image_path)

    # define your bank of kernels
    kernels = {
        'horizontal_edge': np.array([[ 1,  1,  1],
                                     [ 0,  0,  0],
                                     [-1, -1, -1]]),
        'vertical_edge':   np.array([[ 1,  0, -1],
                                     [ 1,  0, -1],
                                     [ 1,  0, -1]]),
        'diag_down':       np.array([[ 1,  1,  0],
                                     [ 1,  0, -1],
                                     [ 0, -1, -1]]),
        'diag_up':         np.array([[ 0,  1,  1],
                                     [-1,  0,  1],
                                     [-1, -1,  0]]),
        'laplacian':       np.array([[ 0, -1,  0],
                                     [-1,  4, -1],
                                     [ 0, -1,  0]]),
        'sharpen':         np.array([[ 0, -1,  0],
                                     [-1,  5, -1],
                                     [ 0, -1,  0]]),
        'box_blur':        np.ones((3,3)) / 9.0
    }

    # for each kernel: convolve and plot
    for name, kernel in kernels.items():
        fmap = apply_fixed_convolution(binary_img, kernel)

        plt.figure(figsize=(4,2))
        # original
        plt.subplot(1,2,1)
        plt.imshow(binary_img, cmap='gray')
        plt.title("Binarized")
        plt.axis('off')

        # feature map
        plt.subplot(1,2,2)
        plt.imshow(fmap, cmap='gray')
        plt.title(name)
        plt.axis('off')

        plt.tight_layout()

    plt.show()











# fixed_convolution.py

# … your read_and_binarize_image()
# … your kernels = { … }
# … your apply_fixed_convolution()

def combine_feature_maps(feature_maps):
    flat = [ fmap.flatten() for fmap in feature_maps ]
    return np.concatenate(flat)

if __name__ == "__main__":
    img = read_and_binarize_image("…/images/2_0.png")
    feature_maps = [ apply_fixed_convolution(img, k) for k in kernels.values() ]
    feature_vector = combine_feature_maps(feature_maps)

    print("Number of maps:", len(feature_maps))
    print("Feature vector shape:", feature_vector.shape)  # (7*17*17,)
