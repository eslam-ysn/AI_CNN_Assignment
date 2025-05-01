import os
import numpy as np
import matplotlib.pyplot as plt
from fixed_convolution import read_and_binarize_image, kernels, apply_fixed_convolution, combine_feature_maps
from neural_network import NeuralNetwork

# --------------------
# Main: Load trained model, run inference on new images
# --------------------

def load_model(params_path: str, input_size: int, hidden_size: int, output_size: int) -> NeuralNetwork:
    """
    Load network parameters from .npz and return a NeuralNetwork with those weights.
    """
    data = np.load(params_path)
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.W1 = data['W1']
    nn.b1 = data['b1']
    nn.W2 = data['W2']
    nn.b2 = data['b2']
    return nn


def predict_image(nn: NeuralNetwork, image_path: str) -> int:
    """
    Read, process, and predict class index for a single image.
    """
    # 1) Read & binarize
    bin_img = read_and_binarize_image(image_path)
    # 2) Feature extraction
    fmap_list = [apply_fixed_convolution(bin_img, k) for k in kernels.values()]
    fv = combine_feature_maps(fmap_list).reshape(-1, 1)  # shape (D_in,1)
    # 3) Forward pass
    probs, _ = nn.forward(fv)
    # 4) Return predicted class index
    return int(np.argmax(probs))


if __name__ == "__main__":
    # Adjust these sizes to match your training setup
    num_kernels = len(kernels)
    map_size = next(iter(kernels.values())).shape[0]
    D_in = num_kernels * (map_size - 2) * (map_size - 2)  # 3x3 kernels shrink by 2 per dim
    H = 64  # must match training
    D_out = len(kernels)  # or your actual number of classes (e.g., 9 or 10)

    # Load trained model parameters
    model_path = 'model_params.npz'
    nn = load_model(model_path, D_in, H, D_out)

    # Directory of test images
    test_dir = '../images'
    images = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    # Run predictions
    for img_file in images:
        img_path = os.path.join(test_dir, img_file)
        pred = predict_image(nn, img_path)
        print(f"Image {img_file} -> Predicted class: {pred}")

        # Optionally display each image and title
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred}")
        plt.axis('off')
        plt.show()
