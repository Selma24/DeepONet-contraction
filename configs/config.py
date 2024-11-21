import os

# Global settings for model parameters

local_folder = 'path_to_your_local_folder'  # Update this with your actual local folder path

data_folder = os.path.join(local_folder, 'data/100 days prediction') # Make sure "data" is inside your local folder
data_folder_train = os.path.join(data_folder, 'training')
data_folder_eval = os.path.join(data_folder, 'evaluation')

model_folder = os.path.join(local_folder, 'model') 

# For the training files
file_versions = [1, 2, 3, 4, 5]
file_template = '150_FEM_samples_version_{}.mat'

trunk_input_size = 7        
branch_input_size = 5       
hidden_size = [50, 50, 50]  
output_size_branch = 100            
output_size_trunk = 50

learning_rate = 0.001
bs = 100
number_of_epochs = 100 