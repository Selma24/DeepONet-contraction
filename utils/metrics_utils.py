import numpy as np
from shapely.geometry import Polygon

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
