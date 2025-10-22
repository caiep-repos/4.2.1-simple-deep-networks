import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def get_data():
    """
    Loads and preprocesses the Fashion MNIST dataset.
    Returns training and test data as PyTorch tensors.
    """
    # Download and load Fashion MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Extract images and labels
    train_images = train_dataset.data.float() / 255.0
    train_labels = train_dataset.targets
    test_images = test_dataset.data.float() / 255.0
    test_labels = test_dataset.targets

    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    """
    Builds a simple sequential neural network model.

    TODO: Implement this function

    Your model should have:
    - A Flatten layer to convert 28x28 images to 784-dimensional vectors
    - A hidden layer with 128 neurons and ReLU activation
    - An output layer with 10 neurons (one for each class)

    Hint: Use nn.Sequential with nn.Flatten(), nn.Linear(), nn.ReLU()
    """
    # Your code here
    pass

def train_model(model, train_images, train_labels, epochs=5, batch_size=32):
    """
    Compiles and trains the model.

    TODO: Implement this function

    Steps:
    1. Create a DataLoader from the training data
    2. Define a loss function (CrossEntropyLoss)
    3. Define an optimizer (Adam)
    4. Implement the training loop:
       - For each epoch:
         - For each batch:
           - Zero the gradients
           - Forward pass
           - Calculate loss
           - Backward pass
           - Update weights

    Args:
        model: PyTorch model to train
        train_images: Training images tensor
        train_labels: Training labels tensor
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained model
    """
    # Your code here
    pass

def evaluate_model(model, test_images, test_labels):
    """
    Evaluates the model on the test set.

    TODO: Implement this function

    Steps:
    1. Create a DataLoader from the test data
    2. Define a loss function
    3. Set model to evaluation mode
    4. Iterate through test data (without gradients):
       - Calculate predictions
       - Calculate loss
       - Calculate accuracy

    Args:
        model: Trained PyTorch model
        test_images: Test images tensor
        test_labels: Test labels tensor

    Returns:
        test_loss: Average test loss (float)
        test_acc: Test accuracy as percentage (float)
    """
    # Your code here
    pass
