import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import matplotlib.axes as ax
plt.rcParams.update({'font.size': 21})
from shapely.geometry import Polygon

def load_matlab_data(filename):
  mat_data = scipy.io.loadmat(filename)
  input_data = mat_data['input_data']                                   
  time_steps = mat_data['time_steps'][0]
  displacement = mat_data['p_displ'][0]                 	              
  initial_coordinates = mat_data['initial_p_coordinates'][0]            
  wound_displacement = mat_data['wound_displ'][0]                       
  initial_wound_coordinates = mat_data['initial_wound_coordinates'][0]  
  initial_shape_info = mat_data['initial_shape_info']                   
  domain_sizes = mat_data['domain_sizes']                               

  return input_data, time_steps, displacement, initial_coordinates, wound_displacement, initial_wound_coordinates, initial_shape_info, domain_sizes

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk):
        super(DeepONet, self).__init__()
        self.trunk_net = MLP(trunk_input_size, hidden_size, output_size_trunk)
        self.branch_net = MLP(branch_input_size, hidden_size, output_size_branch)

    def forward(self, x_trunk, x_branch, domain_sizes):
        output_trunk = self.trunk_net(x_trunk)

        output_branch = self.branch_net(x_branch)

        output1 = torch.sum(torch.mul(output_trunk, output_branch[:, :50]), dim=1, keepdim=True) # this is u_x
        output2 = torch.sum(torch.mul(output_trunk, output_branch[:, 50:]), dim=1, keepdim=True) # this is u_y

        # We modify the outputs with the sine functions
        output1 = output1 * self.sine(x_trunk[:,1], domain_sizes[:,0]) * self.cosine(x_trunk[:,2], domain_sizes[:,1])
        output2 = output2 * self.sine(x_trunk[:,2], domain_sizes[:,1]) * self.cosine(x_trunk[:,1], domain_sizes[:,0])

        return torch.cat((output1, output2), dim=1)

    def sine(self, x, size_dom):
      return torch.sin((np.pi*x.unsqueeze(1))/size_dom.unsqueeze(1))

    def cosine(self, x, size_dom):
      return torch.cos((np.pi*x.unsqueeze(1))/(2*size_dom.unsqueeze(1)))
    
def prepare_test_data_per_sample(input_FEM, output_FEM, coordinates, time_steps, domain_info, initial_shape_info, num_samples_in_domain):
  num_time_samples = len(output_FEM[0])

  input_to_branch = np.repeat(np.expand_dims(input_FEM, axis=0), repeats=num_samples_in_domain*num_time_samples, axis=0)
  input_for_sine = np.repeat(np.expand_dims(domain_info, axis=0), repeats=num_samples_in_domain*num_time_samples, axis=0)
  input_to_trunk = np.zeros((len(input_to_branch),7))
  truth_values = np.zeros((len(input_to_branch),2))

  indices_time = np.arange(num_time_samples)
  indices_space = np.arange(num_samples_in_domain)

  for p in range(num_time_samples):
    input_to_trunk[p*num_samples_in_domain:(p+1)*num_samples_in_domain,0] = np.repeat(time_steps[0][indices_time[p]], repeats=num_samples_in_domain, axis=0)
    input_to_trunk[p*num_samples_in_domain:(p+1)*num_samples_in_domain,1:3] = coordinates[indices_space]
    truth_values[p*num_samples_in_domain:(p+1)*num_samples_in_domain,:] = output_FEM[0][indices_time[p]][indices_space]

  input_to_trunk[:,3:7] = np.repeat(np.expand_dims(initial_shape_info, axis=0), repeats=num_samples_in_domain*num_time_samples, axis=0)

  return input_to_branch, input_to_trunk, truth_values, input_for_sine

def evaluate_model_on_test_sample(test_set):
  model.eval()

  test_losses = []
  predicted_values = []

  with torch.no_grad():
      for i in range(len(test_set)):
          trunk_data, branch_data, truth_data, sine_data = test_set[i]  # Get individual data point
          trunk_data = trunk_data.unsqueeze(0)  # Add batch dimension
          branch_data = branch_data.unsqueeze(0)  # Add batch dimension
          truth_data = truth_data.unsqueeze(0)  # Add batch dimension
          sine_data = sine_data.unsqueeze(0)

          output_test = model(trunk_data, branch_data, sine_data)  # Forward pass

          # Calculate loss
          loss_test = criterion(output_test, truth_data)
          test_losses.append(loss_test.item())

          # Convert predictions and true values to numpy arrays for comparison
          predicted_values.append(output_test.numpy())

  # Concatenate predicted values into a numpy array
  predicted_values = np.concatenate(predicted_values, axis=0)

  # Calculate total test loss
  total_test_loss = np.sum(test_losses)

  return test_losses, predicted_values, total_test_loss

def calculate_rsaw(input_to_trunk, truth_values, predicted_values, num_sampled_coord, iter):
  boundary_coordinates_start = input_to_trunk[iter*num_sampled_coord:(iter+1)*num_sampled_coord,1:3]
  boundary_coordinates_start_incl_0 = np.zeros((num_sampled_coord+1,2))
  boundary_coordinates_start_incl_0[0:num_sampled_coord,:] = boundary_coordinates_start

  polygon_start = Polygon(boundary_coordinates_start_incl_0)
  area_start = polygon_start.area

  portion_true_displacements = truth_values[iter*num_sampled_coord:(iter+1)*num_sampled_coord,:]
  boundary_coordinates_deformed_truth = boundary_coordinates_start + portion_true_displacements

  boundary_coordinates_incl_0_truth = np.zeros((num_sampled_coord+1,2))
  boundary_coordinates_incl_0_truth[0:num_sampled_coord,:] = boundary_coordinates_deformed_truth

  polygon_truth = Polygon(boundary_coordinates_incl_0_truth)
  x, y = polygon_truth.exterior.xy

  area_truth = polygon_truth.area

  portion_predicted_displacements = predicted_values[iter*num_sampled_coord:(iter+1)*num_sampled_coord,:]
  boundary_coordinates_deformed_predicted = boundary_coordinates_start + portion_predicted_displacements

  boundary_coordinates_incl_0_pred = np.zeros((num_sampled_coord+1,2))
  boundary_coordinates_incl_0_pred[0:num_sampled_coord,:] = boundary_coordinates_deformed_predicted
  polygon_pred = Polygon(boundary_coordinates_incl_0_pred)

  area_pred = polygon_pred.area

  rsaw_truth = area_truth/area_start
  rsaw_pred = area_pred/area_start

  return rsaw_truth, rsaw_pred

def aR2(truth, prediction):
  M = np.zeros((1,2))
  for i in range(2):
    numerator = np.sum((truth[:,i] - prediction[:,i]) ** 2)
    denominator = np.sum((truth[:,i] - np.mean(truth[:,i])) ** 2)
    M[0,i] = 1 - (numerator / denominator)
  return np.mean(M)

def aRRMSE(truth, prediction):
  M = np.zeros((1,2))
  for i in range(2):
    numerator = np.mean((truth[:,i]-prediction[:,i]) ** 2)
    denominator = np.mean((truth[:,i] - np.mean(truth[:,i])) ** 2)
    M[0,i] = np.sqrt(numerator/denominator)
  return np.mean(M)

def aMARE(truth, prediction):
  truth = np.round(truth, 1)
  prediction = np.round(prediction, 1)
  M = np.zeros((len(truth),2))
  M[truth[:,0] != 0, 0] = np.abs((truth[truth[:,0] != 0, 0] - prediction[truth[:,0] != 0, 0])/truth[truth[:,0] != 0, 0])
  M[truth[:,1] != 0, 1] = np.abs((truth[truth[:,1] != 0, 1] - prediction[truth[:,1] != 0, 1])/truth[truth[:,1] != 0, 1])
  return np.mean(np.mean(M, axis=0))

# Define the directories and load test data
local_folder = 'path_to_your_local_folder'  # Update this with your actual local folder path
data_folder = os.path.join(local_folder, 'data')

input_data_test, time_steps_test, displacement_test, initial_coordinates_test, wound_displacement_test, initial_wound_coordinates_test, initial_shape_info_test, domain_sizes_test = load_matlab_data(os.path.join(data_folder, '150_FEM_convex.mat'))

# Initialize trained model
trunk_input_size = 7        
branch_input_size = 5       
hidden_size = [50, 50, 50]  
output_size_branch = 100            
output_size_trunk = 50

model = DeepONet(trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk)
model.load_state_dict(torch.load(os.path.join(local_folder, 'time10_coord20_bs100_epochs100.pth')))

criterion = nn.MSELoss()

# Evaluate model on chosen sample from the test set
total_test_losses = []
rsaws_truth = []

for sample in range(0,1):
  input_to_branch_test, input_to_trunk_test, truth_values_test, input_to_sine_test = prepare_test_data_per_sample(input_data_test[sample], displacement_test[sample], initial_coordinates_test[sample], time_steps_test[sample], domain_sizes_test[sample], initial_shape_info_test[sample], len(initial_coordinates_test[sample]))

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

# Evaluate trained model on the wound boundary (not nodes in the grid)
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

# Evaluate on entire test set. Save all truth values and predictions for all samples, for all time
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