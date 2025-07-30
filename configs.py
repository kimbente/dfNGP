################
### SIM DATA ###
################

# directories (alphabetic order)
dfGP_SIM_RESULTS_DIR = "results_sim/dfGP"
dfGPcm_SIM_RESULTS_DIR = "results_sim/dfGPcm"
dfNGP_SIM_RESULTS_DIR = "results_sim/dfNGP"
dfNN_SIM_RESULTS_DIR = "results_sim/dfNN"
GP_SIM_RESULTS_DIR = "results_sim/GP"
PINN_SIM_RESULTS_DIR = "results_sim/PINN"

# learning rates (alphabetic order)
# NOTE: df is always smallcap.
# NOTE: All lr are the same for SIM
dfGP_SIM_LEARNING_RATE = 0.005
dfGPcm_SIM_LEARNING_RATE = 0.005 
dfNGP_SIM_LEARNING_RATE = 0.005 # lr x 0.02 for NN mean function params
dfNN_SIM_LEARNING_RATE = 0.005
GP_SIM_LEARNING_RATE = 0.005
PINN_SIM_LEARNING_RATE = 0.005

# SIM-specific hyperparameters
# test grid resolution
N_SIDE = 20

#################
### REAL DATA ###
#################

# directories (alphabetic order)
dfGP_REAL_RESULTS_DIR = "results_real/dfGP"
dfGPcm_REAL_RESULTS_DIR = "results_real/dfGPcm"
dfNGP_REAL_RESULTS_DIR = "results_real/dfNGP"
dfNN_REAL_RESULTS_DIR = "results_real/dfNN"
GP_REAL_RESULTS_DIR = "results_real/GP"
PINN_REAL_RESULTS_DIR = "results_real/PINN"

# learning rates (alphabetic order)
dfGP_REAL_LEARNING_RATE = 0.005
dfGPcm_REAL_LEARNING_RATE = 0.005
dfNGP_REAL_LEARNING_RATE = 0.005 # lr x 0.2 for NN mean function params
dfNN_REAL_LEARNING_RATE = 0.005 
GP_REAL_LEARNING_RATE = 0.005
# NOTE: PINN requires slightly lower lr for smooth descent on real train, otherwise early stopping triggers soon
PINN_REAL_LEARNING_RATE = 0.001

# infer at higher resolution grid across the domain 
# NOTE: for visualisations only, not evaluations
N_SIDE_INFERENCE = 60

################################
### TRAINING HYPERPARAMETERS ###
################################

# Toggle emission tracking with codecarbon on or off
# TRACK_EMISSIONS_BOOL = False
TRACK_EMISSIONS_BOOL = True

# Define how often to print training progress
PRINT_FREQUENCY = 50

NUM_RUNS = 8
MAX_NUM_EPOCHS = 2000

PATIENCE = 100 # Stop after {PATIENCE} epochs with no improvement
GP_PATIENCE = 50 # NOTE: GP convergence is more smooth so less patience is needed

# WEIGHT_DECAY is L2 regularisation; `decay` because it pulls weights towards 0
# Only for NN parameters, not for GP parameters
WEIGHT_DECAY = 1e-4 # i.e. 0.0001

BATCH_SIZE = 32

# PINN HYPERPARAM (SIM & REAL)
W_PINN_DIV_WEIGHT = 0.3

# FOR SIM
# Noise parameter for training: independent Gaussian noise to perturb inputs
# NOTE: This corresponds to a true noise variance of 0.0004 (contained in initialisation range)
# We do scale by the var of each experiment however (which is slighly <1 or >1)
STD_GAUSSIAN_NOISE = 0.02

##########################################
### DEFAULT/SIM (df)GP HYPERPARAMETERS ###
##########################################
# order: lengthscale, outputscale variance, noise variance

# HYPERPARAMETER 1: Range for lengthscale parameter (l) 
# NOTE: This corresponds to a l^2 range of (0.09, 0.49) (domain is [0, 1])
L_RANGE = (0.3, 0.7)

# HYPERPARAMETER 2: Range for outputscale variance parameter (sigma_f^2)
# NOTE: This corresponds to a sigma_f range of (0.64, ~1.22)
OUTPUTSCALE_VAR_RANGE = (0.8, 1.5)

# NOTE: For residual models (i.e. models with non-zero mean function), we use a different range for the outputscale variance, acknowledging that the residuals are smaller than the original data.
OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE = (0.1, 0.6)

# For regular GP only, we scale for each task
# NOTE: The multitask GP is parameterised via a covariance factor F, which is used to construct the covariance matrix B together with a TASK Variance D.
# B = (FF^T + D), where D is a diagonal matrix and F is the covar_factor
TASK_COVAR_FACTOR_RANGE = (-0.2, 0.5) 

# Define initialisation ranges FOR GP MODELs
# HYPERPARAMETER 3: Range for noise variance parameter (sigma_n^2)
# NOTE: This corresponds to a sigma_n range of (0.01, 0.07)
NOISE_VAR_RANGE = (0.0001, 0.0049)

###################################
### REAL (df)GP HYPERPARAMETERS ###
###################################

# Scale input bacl to ~km
SCALE_INPUT_region_lower_byrd = 30
SCALE_INPUT_region_mid_byrd = 70
SCALE_INPUT_region_upper_byrd = 70

# NOTE: This corresponds to a l^2 range of (25.0, 64.0) (domain is e.g. [0, 70])
REAL_L_RANGE = (5.0, 8.0)
REAL_NOISE_VAR_RANGE = (0.01, 0.05)
REAL_OUTPUTSCALE_VAR_RANGE = (1.0, 1.8) # successful
REAL_OUTPUTSCALE_VAR_RANGE = (0.8, 1.4)