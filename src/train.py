import os
import numpy as np
from fixed_convolution import read_and_binarize_image, kernels, apply_fixed_convolution, combine_feature_maps
from neural_network import NeuralNetwork

# --------------------
# 1) Load and prepare dataset
# --------------------
image_dir = "../images"  # adjust path to your images folder
# collect .png files
all_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
# full paths and labels inferred from filename (e.g. '2_0.png' -> class 2)
image_paths = [os.path.join(image_dir, f) for f in all_files]
labels      = [int(f.split('_')[0]) for f in all_files]

# shuffle
indices = np.arange(len(image_paths))
np.random.shuffle(indices)
image_paths = [image_paths[i] for i in indices]
labels      = [labels[i] for i in indices]

# convert each image to feature-vector
data = []
for path, label in zip(image_paths, labels):
    bin_img = read_and_binarize_image(path)
    fmap_list = [apply_fixed_convolution(bin_img, k) for k in kernels.values()]
    fv = combine_feature_maps(fmap_list).reshape(-1, 1)  # shape (D_in, 1)
    data.append((fv, label))

# split into train/test (80/20)
split = int(0.8 * len(data))
train_data = data[:split]
test_data  = data[split:]

# --------------------
# 2) Initialize neural network
# --------------------
D_in  = train_data[0][0].shape[0]          # e.g. num_kernels*17*17
H     = 64                                # hidden units
D_out = len(set(labels))                  # number of classes
nn    = NeuralNetwork(D_in, H, D_out)

# --------------------
# 3) Training loop
# --------------------
learning_rate = 0.01
num_epochs    = 50

for epoch in range(1, num_epochs+1):
    total_loss = 0
    # training
    for x, lbl in train_data:
        # one-hot encode label
        y = np.zeros((D_out, 1))
        y[lbl, 0] = 1
        # forward/backward/update
        A2, cache = nn.forward(x)
        total_loss += nn.compute_loss(A2, y)
        grads = nn.backward(cache, y)
        nn.update_params(grads, learning_rate)
    avg_loss = total_loss / len(train_data)

    # evaluation on test set
    correct = 0
    for x_test, lbl_test in test_data:
        A2_test, _ = nn.forward(x_test)
        if np.argmax(A2_test) == lbl_test:
            correct += 1
    accuracy = correct / len(test_data)

    print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2%}")

# Optionally save model parameters
np.savez('model_params.npz', W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)
print("Training complete. Model saved to 'model_params.npz'.")
