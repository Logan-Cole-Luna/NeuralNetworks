import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from BackPropNetwork import NeuralNet, tanh, relu, mean_squared_error

MNIST_DIR = "mnist"
CACHE_DIR = "cached_data"
MNIST_DATA_FILE = "mnist_data.npz"

def load_local_mnist():
    """Load MNIST from local directory"""
    print("Loading MNIST from local files...")
    
    # Load training data
    with open(os.path.join(MNIST_DIR, "train-images-idx3-ubyte/train-images-idx3-ubyte"), 'rb') as f:
        # Skip header bytes
        f.read(16)
        train_images = np.frombuffer(f.read(), dtype=np.uint8)
        train_images = train_images.reshape(-1, 28*28)
    
    with open(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"), 'rb') as f:
        # Skip header bytes
        f.read(8)
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return train_images, train_labels

def load_mnist_data():
    """Load and preprocess MNIST dataset with caching"""
    cache_path = os.path.join(CACHE_DIR, MNIST_DATA_FILE)
    
    # Try to load cached data first
    if os.path.exists(cache_path):
        print("Loading cached MNIST dataset...")
        with np.load(cache_path) as data:
            return data['scaled_data'], data['encoded_targets']
    
    # Load from local files
    data, labels = load_local_mnist()
    
    # Process the data
    print("Processing MNIST data...")
    
    # One-hot encode targets
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_targets = encoder.fit_transform(labels.reshape(-1,1)).toarray()
    
    # Scale data to [-1,1] range
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(data)
    encoded_targets = scaler.fit_transform(encoded_targets)
    
    # Save processed data to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Saving processed data to cache...")
    np.savez_compressed(
        cache_path, 
        scaled_data=scaled_data, 
        encoded_targets=encoded_targets
    )
    
    return scaled_data, encoded_targets

def plot_digit(image_data):
    """Plot a single MNIST digit"""
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def create_mini_dataset(scaled_data, encoded_targets, train_size=300, test_size=25):
    """Create smaller training and test sets"""
    X_train = scaled_data[:train_size]
    y_train = encoded_targets[:train_size]
    
    X_test = scaled_data[train_size:train_size+test_size]
    y_test = encoded_targets[train_size:train_size+test_size]
    
    return X_train, y_train, X_test, y_test

def main():
    # Load and preprocess data
    try:
        scaled_data, encoded_targets = load_mnist_data()
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        return
    
    X_train, y_train, X_test, y_test = create_mini_dataset(scaled_data, encoded_targets)
    
    # Simplified network topology for MNIST
    network_topology = [784, 128, 64, 10]  # Smaller network, still deep enough
    
    # Create and train network with optimized parameters
    nnet = NeuralNet(
        topology=network_topology,
        learning_rate=0.1,      # Increased learning rate
        momentum=0.9,           # Increased momentum
        init_method='xavier',
        hidden_activation_func=relu,  # ReLU for hidden layers
        output_activation_func=tanh   # tanh for output layer
    )
    
    print(f"Network topology: {nnet.shape}")
    print(f"Trainable parameters: {nnet.n_trainable_params}")
    
    # Train with online learning instead of batch
    error = nnet.train(
        X_train, 
        y_train, 
        epochs=150,
        batch_size=0,  # Changed to online learning
        error_threshold=1e-3,  # More realistic threshold
        verbose=True
    )
    
    # Test the network
    print("\nTesting network on sample digits:")
    for i in range(5):  # Test first 5 digits
        _x = X_test[[i]]
        _y = y_test[[i]]
        
        _learned_y = nnet.feedforward(_x)
        sample_error = mean_squared_error(_y, _learned_y)
        
        plt.figure(figsize=(3,3))
        plot_digit(_x[0])
        plt.title(f"Predicted: {np.argmax(_learned_y)}\nActual: {np.argmax(_y)}")
        plt.show()
        print(f"Sample {i} error: {sample_error:.4f}")

if __name__ == "__main__":
    main()
