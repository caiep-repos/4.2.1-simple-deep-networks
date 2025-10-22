# Simple Deep Network for Image Classification

## Problem Description

In this assignment, you will build, train, and evaluate a simple deep neural network for image classification using **PyTorch**. You will use the Fashion MNIST dataset, which consists of 28x28 grayscale images of 10 different types of clothing.

## Learning Objectives

- Understand the basics of building a neural network with PyTorch
- Learn how to define a sequential model with linear layers
- Compile and train a model using an optimizer and loss function
- Evaluate the performance of your trained model

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

## Instructions

1. Open the `assignment.py` file.
2. You will find three function definitions. Your task is to implement them.
   * **Task 1**: Implement the `build_model` function to create a PyTorch Sequential model. The model should have an input layer (Flatten), at least one hidden layer (128 units with ReLU), and an output layer (10 units).
   * **Task 2**: Implement the `train_model` function to compile and train the model on the training data. Use CrossEntropyLoss and Adam optimizer.
   * **Task 3**: Implement the `evaluate_model` function to evaluate the trained model on the test data. Return loss (float) and accuracy (float percentage).

## Testing Your Solution

Run the test file to verify your implementation:

```bash
python -m unittest test
```

## Expected Results

After training for 5 epochs:
- Training accuracy: ~88-90%
- Test accuracy: ~85-87%

## Need Help?

Check `sample_submission.py` for a complete working solution.

## Model Architecture

Your model should have the following structure:
```
Input (28×28) → Flatten (784) → Dense(128, ReLU) → Dense(10) → Output
```

## Tips

1. Use `nn.Sequential()` to build your model
2. Remember to flatten the images: `images.view(-1, 28 * 28)`
3. Return accuracy as a percentage (0-100), not a decimal
4. Don't forget to return the trained model from `train_model()`

Good luck! 🚀
