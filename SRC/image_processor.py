import numpy as np
import matplotlib.pyplot as plt
import os
import math

def load_images(path):
    """
    Load images from the specified path using matplotlib.
    
    Args:
        path (str): Path to the directory containing images
        
    Returns:
        tuple: (data, labels) where data is a numpy array of shape (n, 19, 19)
               and labels is a numpy array of shape (n,)
    """
    # List all PNG files in the directory
    files = [f for f in os.listdir(path) if f.endswith('.png') and '_' in f and 'augmented' not in f]
    
    # Initialize arrays
    data = []
    labels = []
    
    for file in files:
        # Extract label from filename (signnumber_version.png)
        label = int(file.split('_')[0])
        
        # Read image using matplotlib
        img_path = os.path.join(path, file)
        img = plt.imread(img_path)
        
        # Convert to grayscale if it's RGB
        if len(img.shape) > 2:
            img = np.mean(img[:, :, :3], axis=2)  # Convert RGB to grayscale
        
        # Scale to 0-255
        if img.max() <= 1.0:
            img = img * 255
            
        img = img.astype(np.uint8)
        
        # Append to lists
        data.append(img)
        labels.append(label)
    
    # Convert to numpy arrays
    data = np.array(data, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    
    return data, labels

def binarize_image(img, thresh):
    """
    Binarize an image using a threshold.
    
    Args:
        img (numpy.ndarray): Input image
        thresh (int): Threshold value
        
    Returns:
        numpy.ndarray: Binarized image with values 0 and 1
    """
    return (img <= thresh).astype(np.uint8)  # Inverted comparison

def visualize_samples(imgs, labels):
    """
    Visualize random samples from the dataset.
    
    Args:
        imgs (numpy.ndarray): Array of images
        labels (numpy.ndarray): Array of labels
    """
    # Select 5 random indices
    n_samples = len(imgs)
    indices = np.random.choice(n_samples, 5, replace=False)
    
    # Create a figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        axes[i].imshow(imgs[idx], cmap='binary')
        axes[i].set_title(f"Label: {labels[idx]}")
        axes[i].axis('off')  # Turn off axes for clarity
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SRC', 'sample_visualization.png'))
    plt.close()

def rotate_image(img, angle):
    """
    Rotate an image by a given angle using manual inverse mapping.
    
    Args:
        img (numpy.ndarray): Input image
        angle (float): Rotation angle in degrees
        
    Returns:
        numpy.ndarray: Rotated image
    """
    height, width = img.shape
    center_y, center_x = height // 2, width // 2
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    # Create output image
    rotated = np.zeros_like(img)
    
    # Inverse mapping
    for y_out in range(height):
        for x_out in range(width):
            # Translate to origin
            y_centered = y_out - center_y
            x_centered = x_out - center_x
            
            # Apply inverse rotation
            y_in = int(y_centered * cos_theta + x_centered * sin_theta + center_y)
            x_in = int(-y_centered * sin_theta + x_centered * cos_theta + center_x)
            
            # Check if the point is within bounds
            if 0 <= y_in < height and 0 <= x_in < width:
                rotated[y_out, x_out] = img[y_in, x_in]
    
    return rotated

def translate_image(img, dx, dy):
    """
    Translate an image by dx, dy pixels with zero padding.
    
    Args:
        img (numpy.ndarray): Input image
        dx (int): Translation in x direction
        dy (int): Translation in y direction
        
    Returns:
        numpy.ndarray: Translated image
    """
    height, width = img.shape
    translated = np.zeros_like(img)
    
    for y in range(height):
        for x in range(width):
            y_in = y - dy
            x_in = x - dx
            
            if 0 <= y_in < height and 0 <= x_in < width:
                translated[y, x] = img[y_in, x_in]
    
    return translated

def flip_image(img, horizontal=True):
    """
    Flip an image horizontally or vertically.
    
    Args:
        img (numpy.ndarray): Input image
        horizontal (bool): If True, flip horizontally, else flip vertically
        
    Returns:
        numpy.ndarray: Flipped image
    """
    height, width = img.shape
    flipped = np.zeros_like(img)
    
    for y in range(height):
        for x in range(width):
            if horizontal:
                flipped[y, x] = img[y, width - 1 - x]
            else:
                flipped[y, x] = img[height - 1 - y, x]
    
    return flipped

def augment_image(img, label, index):
    """
    Augment an image with rotations, translations, and flips.
    
    Args:
        img (numpy.ndarray): Input image
        label (int): Image label
        index (int): Original image index
        
    Returns:
        list: List of augmented images
    """
    augmented_images = []
    
    # Define augmentation parameters
    angles = [-6, -4, -2, 2, 4, 6]  # Smaller rotation increments up to 6 degrees
    offsets = [-2, -1, 1, 2]        # Include both 1 and 2 pixel translations
    
    # Rotations
    for angle in angles:
        rotated = rotate_image(img, angle)
        augmented_images.append((rotated, label, f"{label}_{index}_rot{angle}"))
    
    # Translations
    for dx in offsets:
        translated_x = translate_image(img, dx, 0)
        augmented_images.append((translated_x, label, f"{label}_{index}_tx{dx}"))
    
    for dy in offsets:
        translated_y = translate_image(img, 0, dy)
        augmented_images.append((translated_y, label, f"{label}_{index}_ty{dy}"))
    
    # Flips
    flipped_h = flip_image(img, horizontal=True)
    augmented_images.append((flipped_h, label, f"{label}_{index}_fliph"))
    
    flipped_v = flip_image(img, horizontal=False)
    augmented_images.append((flipped_v, label, f"{label}_{index}_flipv"))
    
    return augmented_images

def save_augmented_images(augmented_images, output_path):
    """
    Save augmented images to the specified path.
    
    Args:
        augmented_images (list): List of tuples (image, label, filename)
        output_path (str): Path to save the augmented images
    """
    for img, _, filename in augmented_images:
        save_png(img, os.path.join(output_path, f"{filename}.png"))

def save_png(img, file_path):
    """
    Save a numpy array as a PNG file, ensuring binary values remain 0 or 1.
    
    Args:
        img (numpy.ndarray): Image to save
        file_path (str): Path to save the image
    """
    # Use fixed vmin/vmax to ensure values stay at 0 or 1
    plt.imsave(file_path, img, cmap='binary', vmin=0, vmax=1)

# --- Convolution Start ---
def convolve(img, kernel):
    """
    Convolve an image with a kernel using manual convolution with zero padding.
    
    Args:
        img (numpy.ndarray): Input image
        kernel (numpy.ndarray): Convolution kernel
        
    Returns:
        numpy.ndarray: Convolved image
    """
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate needed padding for the kernel
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Create zero-padded version of the image
    padded = np.zeros((img_height + 2 * pad_height, img_width + 2 * pad_width), dtype=np.float32)
    # Copy original image into center of padded array
    padded[pad_height:pad_height + img_height, pad_width:pad_width + img_width] = img
    
    # Initialize output array
    output = np.zeros_like(img, dtype=np.float32)
    
    # Loop through each pixel in the output image
    for y in range(img_height):
        for x in range(img_width):
            # Extract 3x3 region from padded image
            roi = padded[y:y + kernel_height, x:x + kernel_width]
            
            # Compute sum of element-wise multiplication with kernel
            output[y, x] = np.sum(roi * kernel)
    
    return output
# --- Convolution End ---

def extract_features(imgs, kernels):
    """
    Extract features from images using convolution with kernels.
    
    Args:
        imgs (numpy.ndarray): Array of images
        kernels (list): List of convolution kernels
        
    Returns:
        list: List of feature arrays
    """
    features = []
    
    for img in imgs:
        img_features = []
        
        for kernel in kernels:
            # Apply convolution here
            convolved = convolve(img, kernel)
            
            # Flatten and append to features
            img_features.append(convolved.flatten())
        
        # Concatenate all features for this image
        features.append(np.concatenate(img_features))
    
    return features

def build_feature_matrix(feats):
    """
    Build a feature matrix from a list of feature vectors.
    
    Args:
        feats (list): List of feature vectors
        
    Returns:
        numpy.ndarray: Feature matrix
    """
    return np.vstack(feats)

def main():
    """Main function to process images."""
    # Set parameters
    base_path = os.path.dirname(os.path.dirname(__file__))
    images_path = os.path.join(base_path, 'images')
    augmented_path = os.path.join(images_path, 'augmented')
    threshold = 128  # Threshold for binarization
    
    # Define kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    diag1 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    diag2 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    
    kernels = [sobel_x, sobel_y, diag1, diag2, blur]
    
    # Step 1: Load images
    print("Loading images...")
    data, labels = load_images(images_path)
    print(f"Loaded {len(data)} images")
    
    # Step 2: Binarize images
    print("Binarizing images...")
    binarized_data = np.array([binarize_image(img, threshold) for img in data])
    
    # Step 3: Visualize samples
    print("Visualizing samples...")
    visualize_samples(binarized_data, labels)
    
    # Step 4: Augment data
    print("Augmenting data...")
    all_augmented = []
    for i, (img, label) in enumerate(zip(binarized_data, labels)):
        augmented = augment_image(img, label, i)
        all_augmented.extend(augmented)
    
    # Save augmented images
    print("Saving augmented images...")
    save_augmented_images(all_augmented, augmented_path)
    
    # Prepare augmented data for feature extraction
    augmented_data = np.array([img for img, _, _ in all_augmented])
    augmented_labels = np.array([label for _, label, _ in all_augmented])
    
    # Combine original and augmented data
    combined_data = np.vstack([binarized_data, augmented_data])
    combined_labels = np.concatenate([labels, augmented_labels])
    
    # Step 5: Extract features
    print("Extracting features...")
    features = extract_features(combined_data, kernels)
    
    # Build feature matrix
    feature_matrix = build_feature_matrix(features)
    
    # Save data
    print("Saving data...")
    np.save(os.path.join(base_path, 'SRC', 'data.npy'), combined_data)
    np.save(os.path.join(base_path, 'SRC', 'labels.npy'), combined_labels)
    np.save(os.path.join(base_path, 'SRC', 'feature_matrix.npy'), feature_matrix)
    
    # Print shapes
    print(f"Data shape: {combined_data.shape}")
    print(f"Labels shape: {combined_labels.shape}")
    print(f"Feature matrix shape: {feature_matrix.shape}")

if __name__ == "__main__":
    main()
