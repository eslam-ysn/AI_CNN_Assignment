# src/read_images.py

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_binarize_image(image_path, threshold=128):
    """
    Reads an image, converts it to grayscale, and binarizes it.
    
    Args:
        image_path (str): Path to the image file.
        threshold (int): Threshold value for binarization (default is 128).
    
    Returns:
        np.ndarray: Binarized image (0s and 1s).
    """
    # Open the image
    img = Image.open(image_path)

    # Convert image to grayscale
    img = img.convert('L')

    # Convert the grayscale image to a NumPy array
    img_array = np.array(img)

    # Binarize the image: pixels > threshold -> 1, else -> 0
    binary_img = (img_array > threshold).astype(int)

    return binary_img

if __name__ == "__main__":
    # Test the function manually
    image_name = "../images/2_0.png"  # Update to match your file (use 0_0.png, 1_0.png, etc.)

    if os.path.exists(image_name):
        # Read and binarize the image
        binary_image = read_and_binarize_image(image_name)

        # Display the binarized image
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binarized Image')
        plt.show()

        # Print the binary array
        print("Binarized Image Array:")
        print(binary_image)
    else:
        print(f"Image {image_name} not found!")
