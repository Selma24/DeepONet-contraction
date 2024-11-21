import torch
import numpy as np
from torch.utils.data import DataLoader

def evaluate_model_on_test_sample(model, criterion, test_set, batch_size=50):
    model.eval()  # Set model to evaluation mode

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_losses = []
    predicted_values = []

    with torch.no_grad():  # Disable gradient calculation
        for trunk_data, branch_data, truth_data, sine_data in test_loader:
            # Forward pass
            output_test = model(trunk_data, branch_data, sine_data)

            # Calculate loss
            loss_test = criterion(output_test, truth_data)
            test_losses.append(loss_test.item() * trunk_data.size(0))  # Accumulate the loss for the batch

            # Collect predictions
            predicted_values.append(output_test.numpy())  

    predicted_values = np.concatenate(predicted_values, axis=0)

    total_test_loss = np.sum(test_losses) / len(test_set)

    return test_losses, predicted_values, total_test_loss