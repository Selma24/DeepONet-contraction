# DeepONet for predicting post-burn contraction

Welcome to the repository for the code supporting the preprint [Deep operator network models for predicting post-burn contraction](https://arxiv.org/abs/2411.14555). 

### Folders and Files Explained:

- **`data/`**: Folder dedicated to storing the datasets used for training and evaluation.
- **`configs/`**: Contains **`config.py`**, which defines global settings and parameters.
- **`figs/`**: Folder dedicated to storing figures. Contains example figures from the paper.
- **`models/`**: Folder dedicated to storing saved Torch models. Additionaly contains **`model.py`**, which defines the DeepONet. 
- **`utils/`**: Contains various utility functions for data handling, model evaluation, and metric calculation.
- **`evaluate.py`**: Script for evaluating a trained model saved inside **`models/`**. 
- **`train.py`**: Script for training a DeepONet.

### How to Use
1. Download the accompanying dataset from [4TU.Centre for Research Data](https://data.4tu.nl/datasets/69d1aefc-a01d-4280-8b32-5c8420d9a2a3) and place in main repository.
2. Define the parameter **`local_folder`** inside **`config.py`** and initialize global settings and parameters.
3. Utilize **`train.py`** to train the DeepONet. Specify an appropriate name for the model, which will be saved in the **`models/`** directory.
4. Utilize **`evaluate.py`** to evaluate the trained DeepONet. The script considers evaluation on a single sample, evaluation on the wound boundary, and evaluation on the entire test set.
5. Find the saved plots inside  **`figs/`**. 



  
