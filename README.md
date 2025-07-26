# dfNGP
divergence-free Neural Gaussian Processes 

## Set up the environment

Follow these steps to set up the environment for reproducing our results.

Create a new environment named `dfngp_env` with Python 3.10: 

`conda create -n dfngp_env python=3.10`

Follow the prompts to complete creation. Then activate the environment with:

`conda activate dfngp_env`

To install the CUDA-enabled version of PyTorch, use the appropriate build for your system. Our experiments were run using a GPU with CUDA 12.1, so we install:

`pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

Next, use `pip` to install all required packages from the `requirements.txt` file, as well as their dependencies.

`pip install -r /home/kim/ice_thickness/requirements.txt`

Installation may take a few minutes. 

## Run experiments

## List of files with brief explanantions

- [gpytorch_models.py](gpytorch_models.py) defines all GP-based (probabilistic) models using `gpytorch`. This includes dfNGP, dfGP, dfGPcm, and the regular GP. The divergence-free kernel is contained in this file too. The implementation leverages the `linear_operator` package.
- [NN_models.py](NN_models.py) defined all purely neural network-based models using `torch`. This includes the dfNN and PINN. 
- data
    - **real data** contains train and test data as pytorch tensors for the three regions of Byrd glacier, lower, mid and upper.
    - **sim_data**
        - [x_train_lines_discretised_0to1.pt](data/sim_data/x_train_lines_discretised_0to1.pt) defines the input locations for the simulated data. The corresponding training vectors are generated within the experiment scripts, using the function from [simulate.py](simulate.py).
- [configs.py](configs.py) specifies all hyperparameters like learning rates and number of epochs, and also defines the initialisation ranges for GP hyperparameters. Other settings like carbon tracking or print frequencies can be adjusted here. 
- [utils.py](utils.py) contains utility/helper functions.
- [metrics.py](metrics.py) contains metric functions that were required in addition to those provided by packages like gpytorch.
- [simulate.py](simulate.py) contains all functions to generate simulated divergence-free vector fields from inputs x.
- [requirements.txt](requirements.txt) can be use to create a suitable conda environment to reproduce our experiments. The text file lists all key packages necessary to run our code, including the version specifications we used. The instruction to create this environment are given above. 
- [run_sim_experiments_dfGP.py](run_sim_experiments_dfGP.py) contains the script to run dfGP experiments on simulated data.
    - Results & outputs of these experiments are saved in results_sim/dfGP.
- [run_sim_experiments_dfGPcm.py](run_sim_experiments_dfGPcm.py) contains the script to run dfGPcm experiments on simulated data.
    - Results & outputs of these experiments are saved in results_sim/dfGPcm.
- [run_sim_experiments_dfNGP.py](run_sim_experiments_dfNGP.py) contains the script to run dfNGP experiments on simulated data.
    - Results & outputs of these experiments are saved in results_sim/dfNGP.
- [run_sim_experiments_dfNN.py](run_sim_experiments_dfNN.py) contains the script to run dfNN experiments on simulated data.
    - Results & outputs of these experiments are saved in results_sim/dfNN.
- [run_sim_experiments_GP.py](run_sim_experiments_GP.py) contains the script to run dfNN experiments on simulated data.
    - Results & outputs of these experiments are saved in results_sim/GP.
- [run_sim_experiments_PINN.py](run_sim_experiments_PINN.py) contains the script to run dfNN experiments on simulated data. 
    - Results & outputs of these experiments are saved in results_sim/PINN.


