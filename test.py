import unittest
import torch
import torch.nn as nn
from assignment import get_data, build_model, train_model, evaluate_model

class TestSimpleDeepNetwork(unittest.TestCase):
    def test_network(self):
        (train_images, train_labels), (test_images, test_labels) = get_data()

        model = build_model()
        self.assertIsInstance(model, nn.Module)

        # Check model architecture
        layers = list(model.children())
        self.assertEqual(len(layers), 4)  # Flatten, Linear, ReLU, Linear
        self.assertIsInstance(layers[0], nn.Flatten)
        self.assertIsInstance(layers[1], nn.Linear)
        self.assertEqual(layers[1].out_features, 128)
        self.assertIsInstance(layers[3], nn.Linear)
        self.assertEqual(layers[3].out_features, 10)

        model = train_model(model, train_images, train_labels, epochs=1)  # Only 1 epoch for faster testing

        test_loss, test_acc = evaluate_model(model, test_images, test_labels)

        self.assertGreater(test_acc, 80.0)  # Check for reasonable accuracy

if __name__ == '__main__':
    unittest.main()
