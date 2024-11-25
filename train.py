import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 21})
from utils.data_utils import load_and_concatenate_data, prepare_data_for_train
from models.model import DeepONet
from configs.config import *

# Load and prepare training data
(input_data, time_steps, displacement, initial_coordinates, wound_displacement, 
 initial_wound_coordinates, initial_shape_info, domain_sizes)= load_and_concatenate_data(data_folder_train, file_template, file_versions)    

# Standardly we take 10 time steps and 20 spatial points per sample
# Adjust here if needed
(input_to_branch, input_to_trunk, truth_values, input_to_sine) = prepare_data_for_train(input_data, displacement, initial_coordinates, 
                                                                                        time_steps, domain_sizes, initial_shape_info, 20, 10)

# Uncomment if t=365 data needs to be added to training
# (input_data_t365, time_steps_t365, displacement_t365, initial_coordinates_t365, wound_displacement_t365, 
#  initial_wound_coordinates_t365, initial_shape_info_t365, domain_sizes_t365) = load_and_concatenate_data(data_folder_train_t365, file_template_t365, file_versions_t365)
# input_to_branch_t365, input_to_trunk_t365, truth_values_t365, input_to_sine_t365 = prepare_data_for_train(input_data_t365, displacement_t365, initial_coordinates_t365, 
#                                                                                                           time_steps_t365, domain_sizes_t365, initial_shape_info_t365, 
#                                                                                                           20, 10)
# input_to_branch = np.concatenate((input_to_branch, input_to_branch_t365), axis=0)
# input_to_trunk = np.concatenate((input_to_trunk, input_to_trunk_t365), axis=0)
# truth_values = np.concatenate((truth_values, truth_values_t365), axis=0)
# input_to_sine = np.concatenate((input_to_sine, input_to_sine_t365), axis=0)

# Convert to torch tensors
input_to_branch = torch.tensor(input_to_branch, dtype=torch.float32)
input_to_trunk = torch.tensor(input_to_trunk, dtype=torch.float32)
truth_values = torch.tensor(truth_values, dtype=torch.float32)
input_to_sine = torch.tensor(input_to_sine, dtype=torch.float32) # This is for the sine augmentation step

dataset = TensorDataset(input_to_trunk, input_to_branch, truth_values, input_to_sine)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# Initialize model
model = DeepONet(trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk)

# Load saved model as initialization if necessary
# saved_model = os.path.join(model_folder, 'time10_coord20_bs100_epochs100.pth')
# model.load_state_dict(torch.load(saved_model))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

batch_size = bs
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

num_epochs = number_of_epochs 

train_losses = []
validation_losses = []

for epoch in range(1,num_epochs+1):
    model.train()
    train_loss = 0.0
    for trunk_batch, branch_batch, truth_batch, sine_batch in train_dataloader:
        output_train = model(trunk_batch, branch_batch, sine_batch)
        loss_train = criterion(output_train, truth_batch)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        train_loss += loss_train.item()

    train_loss /= len(train_dataloader)

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for trunk_batch, branch_batch, truth_batch, sine_batch in validation_dataloader:
            output_validation = model(trunk_batch, branch_batch, sine_batch)
            loss_validation = criterion(output_validation, truth_batch)
            validation_loss += loss_validation.item()

    validation_loss /= len(validation_dataloader)

    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

    if epoch == 1 or epoch % 50 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.8f}, Validation Loss: {validation_loss:.8f}')

# Save trained model
torch.save(model.state_dict(), os.path.join(model_folder, 'time10_coord20_bs100_epochs100.pth')) # Change name if needed

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training', linewidth=2.5, color='#117A65')
plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation', linewidth=2.5, color='#FFA971')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(os.path.join(figure_folder, 'training_and_validation_loss.png'))
