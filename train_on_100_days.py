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

def prepare_data(input_FEM, output_FEM, coordinates, time_steps, domain_info, initial_shape_info, num_samples_in_domain, num_time_samples):
  input_to_branch = np.repeat(input_FEM, repeats=num_samples_in_domain*num_time_samples, axis=0)
  input_for_sine = np.repeat(domain_info, repeats=num_samples_in_domain*num_time_samples, axis=0)
  input_to_trunk = np.zeros((len(input_to_branch),7))
  truth_values = np.zeros((len(input_to_branch),2))

  for i in range(len(input_FEM)): 
    indices_time = np.random.randint(0, high=len(time_steps[i][0]), size=(num_time_samples,), dtype=int)
    for p in range(num_time_samples):
      indices_space = np.random.randint(0, high=len(coordinates[i]), size=(num_samples_in_domain,), dtype=int)
      input_to_trunk[p*num_samples_in_domain+i*num_time_samples*num_samples_in_domain:(p+1)*num_samples_in_domain+i*num_time_samples*num_samples_in_domain,0] = np.repeat(time_steps[i][0][indices_time[p]], repeats=num_samples_in_domain, axis=0)
      input_to_trunk[p*num_samples_in_domain+i*num_time_samples*num_samples_in_domain:(p+1)*num_samples_in_domain+i*num_time_samples*num_samples_in_domain,1:3] = coordinates[i][indices_space]
      truth_values[p*num_samples_in_domain+i*num_time_samples*num_samples_in_domain:(p+1)*num_samples_in_domain+i*num_time_samples*num_samples_in_domain,:] = output_FEM[i][0][indices_time[p]][indices_space]

  input_to_trunk[:,3:7] = np.repeat(initial_shape_info, repeats=num_samples_in_domain*num_time_samples, axis=0)

  return input_to_branch, input_to_trunk, truth_values, input_for_sine

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

        # We modify the outputs with the sine/cosine functions
        output1 = output1 * self.sine(x_trunk[:,1], domain_sizes[:,0]) * self.cosine(x_trunk[:,2], domain_sizes[:,1])
        output2 = output2 * self.sine(x_trunk[:,2], domain_sizes[:,1]) * self.cosine(x_trunk[:,1], domain_sizes[:,0])

        return torch.cat((output1, output2), dim=1)

    def sine(self, x, size_dom):
      return torch.sin((np.pi*x.unsqueeze(1))/size_dom.unsqueeze(1))

    def cosine(self, x, size_dom):
      return torch.cos((np.pi*x.unsqueeze(1))/(2*size_dom.unsqueeze(1)))
    
# Define the directories and filenames
local_folder = 'path_to_your_local_folder'  # Update this with your actual local folder path
data_folder = os.path.join(local_folder, 'data')
file_versions = [1, 2, 3, 4, 5]
file_template = '150_FEM_version_{}.mat'

# Initialize variables with the data from the first file
first_file_path = os.path.join(data_folder, file_template.format(file_versions[0]))
input_data, time_steps, displacement, initial_coordinates, wound_displacement, initial_wound_coordinates, initial_shape_info, domain_sizes = load_matlab_data(first_file_path)

# Loop through the remaining files and concatenate data
for version in file_versions[1:]:
    file_path = os.path.join(data_folder, file_template.format(version))
    data = load_matlab_data(file_path)
    
    input_data = np.concatenate((input_data, data[0]), axis=0)
    time_steps = np.concatenate((time_steps, data[1]), axis=None)
    displacement = np.concatenate((displacement, data[2]), axis=None)
    initial_coordinates = np.concatenate((initial_coordinates, data[3]), axis=None)
    wound_displacement = np.concatenate((wound_displacement, data[4]), axis=None)
    initial_wound_coordinates = np.concatenate((initial_wound_coordinates, data[5]), axis=None)
    initial_shape_info = np.concatenate((initial_shape_info, data[6]), axis=0)
    domain_sizes = np.concatenate((domain_sizes, data[7]), axis=0)

# Prepare data for DeepONet
input_to_branch, input_to_trunk, truth_values, input_to_sine = prepare_data(input_data, displacement, initial_coordinates, time_steps, domain_sizes, initial_shape_info, 20, 10)

input_to_branch = torch.tensor(input_to_branch, dtype=torch.float32)
input_to_trunk = torch.tensor(input_to_trunk, dtype=torch.float32)
truth_values = torch.tensor(truth_values, dtype=torch.float32)
input_to_sine = torch.tensor(input_to_sine, dtype=torch.float32) # This is for the sine augmentation step

dataset = TensorDataset(input_to_trunk, input_to_branch, truth_values, input_to_sine)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# Initialize model and train
trunk_input_size = 7        
branch_input_size = 5       
hidden_size = [50, 50, 50]  
output_size_branch = 100            
output_size_trunk = 50

model = DeepONet(trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 100 

train_losses = []
validation_losses = []

for epoch in range(1,num_epochs+1):
    for trunk_batch, branch_batch, truth_batch, sine_batch in train_dataloader:
        output_train = model(trunk_batch, branch_batch, sine_batch)

        loss_train = criterion(output_train, truth_batch)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    with torch.no_grad():
        for trunk_batch, branch_batch, truth_batch, sine_batch in validation_dataloader:

            output_validation = model(trunk_batch, branch_batch, sine_batch)


            loss_validation = criterion(output_validation, truth_batch)

    train_losses.append(loss_train.item())
    validation_losses.append(loss_validation.item())

    if epoch == 1:
      print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Validation Loss: {loss_validation.item():.4f}')
    if epoch % 50 ==0:
      print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Validation Loss: {loss_validation.item():.4f}')

# Save trained model
torch.save(model.state_dict(), os.path.join(local_folder, 'time10_coord20_bs100_epochs100.pth'))

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training', linewidth=2.5, color='#117A65')
plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation', linewidth=2.5, color='#FFA971')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
#plt.title('Training and Validation Losses')
plt.legend()
plt.show()