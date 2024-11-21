import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 21})
from utils.data_utils import load_matlab_data, prepare_test_data_per_sample
from models.model import DeepONet
from utils.evaluate_utils import *
from utils.metrics_utils import *
from configs.config import *
    
# Load testing data
input_data_test, time_steps_test, displacement_test, initial_coordinates_test, wound_displacement_test, initial_wound_coordinates_test, initial_shape_info_test, domain_sizes_test = load_matlab_data(os.path.join(data_folder_eval, '150_FEM_samples_convex_comb.mat'))

# Initialize trained model
model = DeepONet(trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk)
model.load_state_dict(torch.load(os.path.join(model_folder, 'time10_coord20_bs100_epochs100.pth'))) # Name of saved and trained model
criterion = nn.MSELoss()

#########################################################################################################################################################################################################################################################################################################################################
# Evaluate model on chosen sample from the test set 
# Plot the true and predicted coordinates at t=0, t=tmax, t=100

total_test_losses = []
rsaws_truth = []

sample = 0 # Choose a sample in [0,149]

input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[sample], displacement_test[sample], initial_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_coordinates_test[sample]))

# Convert to torch tensors
input_to_branch_test = torch.tensor(input_to_branch_test, dtype=torch.float32)
input_to_trunk_test_torch = torch.tensor(input_to_trunk_test, dtype=torch.float32)
truth_values_test_torch = torch.tensor(truth_values_test, dtype=torch.float32)
input_to_sine_test = torch.tensor(input_to_sine_test, dtype=torch.float32)

test_dataset = TensorDataset(input_to_trunk_test_torch, input_to_branch_test, truth_values_test_torch, input_to_sine_test)

test_losses, predicted_values, total_test_loss = evaluate_model_on_test_sample(test_dataset)

total_test_losses.append(total_test_loss)

# This part evaluates on wound boundary, so that we can determine the time of max contraction (=min RSAW)
input_to_branch_test_w, input_to_trunk_test_w, truth_values_test_w, input_to_sine_test_w = prepare_test_data_per_sample(input_data_test[sample], wound_displacement_test[sample], initial_wound_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_wound_coordinates_test[sample]))

input_to_branch_test_w = torch.tensor(input_to_branch_test_w, dtype=torch.float32)
input_to_trunk_test_torch_w = torch.tensor(input_to_trunk_test_w, dtype=torch.float32)
truth_values_test_torch_w = torch.tensor(truth_values_test_w, dtype=torch.float32)
input_to_sine_test_w = torch.tensor(input_to_sine_test_w, dtype=torch.float32)

test_dataset_w = TensorDataset(input_to_trunk_test_torch_w, input_to_branch_test_w, truth_values_test_torch_w, input_to_sine_test_w)

test_losses_w, predicted_values_w, total_test_loss_w = evaluate_model_on_test_sample(test_dataset_w)

for t in range(len(time_steps_test[sample][0])):
    rsaw_truth, _ = calculate_rsaw(input_to_trunk_test_w, truth_values_test_w, predicted_values_w, len(initial_wound_coordinates_test[sample]), t)
    rsaws_truth.append(rsaw_truth)

coord_in_sample = len(initial_coordinates_test[sample])
tsteps_in_sample = len(time_steps_test[sample][0])
tindex_min_rsaw = rsaws_truth.index(min(rsaws_truth))
t_min_rsaw = time_steps_test[sample][0][tindex_min_rsaw]

# Plot t=0
plt.figure(figsize=(8, 8))
plt.plot(initial_coordinates_test[sample][:,0]+truth_values_test[0*coord_in_sample:1*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+truth_values_test[0*coord_in_sample:1*coord_in_sample][:,1], 'o', label='Target', color='black')
plt.plot(initial_coordinates_test[sample][:,0]+predicted_values[0*coord_in_sample:1*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+predicted_values[0*coord_in_sample:1*coord_in_sample][:,1], 'o', label='Prediction', color='#FFA971')
plt.xlabel('x')
plt.ylabel('y')
plt.title('t = 0')
plt.legend(loc='upper right')
plt.show()

# Plot t=100
plt.figure(figsize=(8, 8))
plt.plot(initial_coordinates_test[sample][:,0]+truth_values_test[(tsteps_in_sample-1)*coord_in_sample:tsteps_in_sample*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+truth_values_test[(tsteps_in_sample-1)*coord_in_sample:tsteps_in_sample*coord_in_sample][:,1], 'o', label='Target', color='black')
plt.plot(initial_coordinates_test[sample][:,0]+predicted_values[(tsteps_in_sample-1)*coord_in_sample:tsteps_in_sample*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+predicted_values[(tsteps_in_sample-1)*coord_in_sample:tsteps_in_sample*coord_in_sample][:,1], 'o', label='Prediction', color='#FFA971')
plt.xlabel('x')
plt.ylabel('y')
plt.title('t = 100 ')
plt.legend(loc='upper right')
plt.show()

# Plot t max contraction
plt.figure(figsize=(8, 8))
plt.plot(initial_coordinates_test[sample][:,0]+truth_values_test[(tindex_min_rsaw-1)*coord_in_sample:tindex_min_rsaw*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+truth_values_test[(tindex_min_rsaw-1)*coord_in_sample:tindex_min_rsaw*coord_in_sample][:,1], 'o', label='Target', color='black')
plt.plot(initial_coordinates_test[sample][:,0]+predicted_values[(tindex_min_rsaw-1)*coord_in_sample:tindex_min_rsaw*coord_in_sample][:,0],initial_coordinates_test[sample][:,1]+predicted_values[(tindex_min_rsaw-1)*coord_in_sample:tindex_min_rsaw*coord_in_sample][:,1], 'o', label='Prediction', color='#FFA971')
plt.xlabel('x')
plt.ylabel('y')
plt.title('t = ' + str(round(t_min_rsaw,2)) + ' (max contraction)')
plt.legend(loc='upper right')
plt.show()

#############################################################################################################################################################################################################################################################################################################################
# Evaluate trained model on the wound boundary (not nodes in the grid)
# Plot the true and predicted RSAW curves for the best and worst prediction

total_test_losses = []

for sample in range(len(input_data_test)):
  input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[sample], wound_displacement_test[sample], initial_wound_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_wound_coordinates_test[sample]))

  input_to_branch_test = torch.tensor(input_to_branch_test, dtype=torch.float32)
  input_to_trunk_test_torch = torch.tensor(input_to_trunk_test, dtype=torch.float32)
  truth_values_test_torch = torch.tensor(truth_values_test, dtype=torch.float32)
  input_to_sine_test = torch.tensor(input_to_sine_test, dtype=torch.float32)

  test_dataset = TensorDataset(input_to_trunk_test_torch, input_to_branch_test, truth_values_test_torch, input_to_sine_test)

  test_losses, predicted_values, total_test_loss = evaluate_model_on_test_sample(test_dataset)

  total_test_losses.append(total_test_loss)

# Plot best and worst prediction in terms of RSAW
index_min_max = [total_test_losses.index(min(total_test_losses)), total_test_losses.index(max(total_test_losses))]

for index in index_min_max:
  input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[index], wound_displacement_test[index], initial_wound_coordinates_test[index], time_steps_test[index], domain_sizes_test[index], initial_shape_info_test[index], len(initial_wound_coordinates_test[index]))

  input_to_branch_test = torch.tensor(input_to_branch_test, dtype=torch.float32)
  input_to_trunk_test_torch = torch.tensor(input_to_trunk_test, dtype=torch.float32)
  truth_values_test_torch = torch.tensor(truth_values_test, dtype=torch.float32)
  input_to_sine_test = torch.tensor(input_to_sine_test, dtype=torch.float32)

  test_dataset = TensorDataset(input_to_trunk_test_torch, input_to_branch_test, truth_values_test_torch, input_to_sine_test)

  test_losses, predicted_values, total_test_loss = evaluate_model_on_test_sample(test_dataset)

  times_rsaw = np.zeros((len(wound_displacement_test[index][0]),3))

  for i in range(len(wound_displacement_test[index][0])):
    rsaw_truth, rsaw_pred = calculate_rsaw(input_to_trunk_test, truth_values_test, predicted_values, len(initial_wound_coordinates_test[index]), i)
    times_rsaw[i,1] = rsaw_truth
    times_rsaw[i,2] = rsaw_pred

    times_rsaw[i,0] = input_to_trunk_test[i*len(initial_wound_coordinates_test[index]),0]

  times_rsaw = times_rsaw[times_rsaw[:, 0].argsort()]

  plt.figure(figsize=(8, 8))
  plt.plot(times_rsaw[:,0],times_rsaw[:,1], label='Target', linewidth=2.5)
  plt.plot(times_rsaw[:,0],times_rsaw[:,2], label='Predicted', linewidth=2.5)
  plt.xlabel('time (days)')
  plt.ylabel('RSAW')
  plt.legend()
  plt.show()

################################################################################################################################################################################################################################################################################################################################################
# Evaluate on entire test set. Save all truth values and predictions for all samples, for all time
# Plot true vs predicted (scatter plots) for entire test set
# Print the performance metrics

sample = 0

input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[sample], displacement_test[sample], initial_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_coordinates_test[sample]))

input_to_branch_test = torch.tensor(input_to_branch_test, dtype=torch.float32)
input_to_trunk_test_torch = torch.tensor(input_to_trunk_test, dtype=torch.float32)
truth_values_test_torch = torch.tensor(truth_values_test, dtype=torch.float32)
input_to_sine_test = torch.tensor(input_to_sine_test, dtype=torch.float32)

test_dataset = TensorDataset(input_to_trunk_test_torch, input_to_branch_test, truth_values_test_torch, input_to_sine_test)

test_losses, predicted_values, total_test_loss = evaluate_model_on_test_sample(test_dataset)

combined_truth_values =  truth_values_test
combined_predicted_values = predicted_values

for sample in range(len(input_data_test)):
  input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[sample], displacement_test[sample], initial_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_coordinates_test[sample]))
  combined_truth_values = np.concatenate((combined_truth_values, truth_values_test), axis=0)

  input_to_branch_test = torch.tensor(input_to_branch_test, dtype=torch.float32)
  input_to_trunk_test_torch = torch.tensor(input_to_trunk_test, dtype=torch.float32)
  truth_values_test_torch = torch.tensor(truth_values_test, dtype=torch.float32)
  input_to_sine_test = torch.tensor(input_to_sine_test, dtype=torch.float32)

  test_dataset = TensorDataset(input_to_trunk_test_torch, input_to_branch_test, truth_values_test_torch, input_to_sine_test)

  test_losses, predicted_values, total_test_loss = evaluate_model_on_test_sample(test_dataset)

  combined_predicted_values = np.concatenate((combined_predicted_values, predicted_values), axis=0)

# Save to a file
np.savez(os.path.join(local_folder, 'combined_truth_and_predicted_values.npz'), combined_truth_values=combined_truth_values, combined_predicted_values=combined_predicted_values)

# Print the performance metrics
print(aR2(combined_truth_values, combined_predicted_values))
print(aRRMSE(combined_truth_values, combined_predicted_values))
print(aMARE(combined_truth_values, combined_predicted_values))

# Make scatter plots with truth vs predicted and y=x
plt.figure(figsize=(8, 8))
plt.scatter(combined_truth_values[:,0], combined_predicted_values[:,0], label='(true, pred)', s=5, color='#73C6B6')
plt.plot(np.linspace(-0.9,0,50),np.linspace(-0.9,0,50),'black', label='y = x')
plt.xlabel('True x-displacement (cm)')
plt.ylabel('Predicted x-displacement (cm)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(combined_truth_values[:,1], combined_predicted_values[:,1], label='(true, pred)', s=5, color='#e88646')
plt.plot(np.linspace(-0.9,0,50),np.linspace(-0.9,0,50),'black', label='y = x')
plt.xlabel('True y-displacement (cm)')
plt.ylabel('Predicted y-displacement (cm)')
plt.legend()
plt.show()