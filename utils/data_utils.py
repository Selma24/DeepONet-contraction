import os
import scipy.io
import numpy as np

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

def load_and_concatenate_data(data_folder, file_template, file_versions):
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

  return input_data, time_steps, displacement, initial_coordinates, wound_displacement, initial_wound_coordinates, initial_shape_info, domain_sizes

def prepare_data_for_train(input_FEM, output_FEM, coordinates, time_steps, domain_info, initial_shape_info, num_samples_in_domain, num_time_samples):
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

