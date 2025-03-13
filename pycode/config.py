from yacs.config import CfgNode as CN
import os

_C = CN()

##################
##### OUTPUT ##### 
##################
_C.OUTPUT = CN()
_C.OUTPUT.BASE_DIR = "../result"
_C.OUTPUT.NUM_TRIAL = 5
_C.OUTPUT.MAX_ITER = 100000
_C.OUTPUT.SAVE_ITER = 10000 # interval to save model and log eval loss
_C.OUTPUT.LOG_ITER = 100 # interval to log training loss
_C.OUTPUT.EVAL_ITER = 1000

###################
##### DATASET ##### 
###################

_C.DATASET = CN()
_C.DATASET.NAME = "RLBench"
_C.DATASET.BATCH_SIZE = 32
_C.DATASET.IMAGE_SIZE = 256

### RLBENCH ###
_C.DATASET.RLBENCH = CN()
_C.DATASET.RLBENCH.TASK_NAME = 'PickUpCup' # e.g. 'CloseJar', 'PickUpCup'
_C.DATASET.RLBENCH.PATH = '../dataset/RLBench-Local' # '../dataset/RLBench-Local'

_C.DATASET.BAXTER = CN()
_C.DATASET.BAXTER.PATH = '../dataset/baxter_demos' # '../dataset/RLBench-Local'

#################
##### NOISE #####
#################

_C.NOISE = CN()
_C.NOISE.METHOD = "gaussian" # gaussian, vae

### GAUSSIAN NOISE ###
_C.NOISE.GAUSSIAN = CN()
_C.NOISE.GAUSSIAN.POSE_RANGE = 0.03
_C.NOISE.GAUSSIAN.ROT_RANGE = 10.0
_C.NOISE.GAUSSIAN.GRASP_PROB = 0.1

### VAE NOISE ###
_C.NOISE.VAE = CN()
_C.NOISE.VAE.STD_MUL = 1.0

####################
##### SAMPLING #####
####################

_C.SAMPLING = CN()
_C.SAMPLING.FIRST_SAMPLE = "random_pick_with_noise" # random_range, random_pick, random_pick_with_noise, vae_noise
_C.SAMPLING.SECOND_SAMPLE = "DFO" # random_all, random_frame, langevin
_C.SAMPLING.INF_SAMPLE = "langevin"
_C.SAMPLING.NUM_NEGATIVE = 64
_C.SAMPLING.SECOND_SAMPLE_STEP = 0
_C.SAMPLING.LIMIT_SAMPLE = 8

### RANDOM_PICK_WITH_NOISE
_C.SAMPLING.RPWN = CN()
_C.SAMPLING.RPWN.RANGE = 0.1
_C.SAMPLING.RPWN.INCLUDE_SELF = True

### vae ###
_C.SAMPLING.VAE = CN()
_C.SAMPLING.VAE.NOISE_STD = 1.0

### langevin ###
_C.SAMPLING.LANGEVIN = CN()
_C.SAMPLING.LANGEVIN.STEP_SIZE = 0.01
_C.SAMPLING.LANGEVIN.MOMENTUM = 0.
_C.SAMPLING.LANGEVIN.ITERATION = 20
_C.SAMPLING.LANGEVIN.DECAY_RATIO = 0.5
_C.SAMPLING.LANGEVIN.DECAY_STEP = 5

### langevin_vae ###
_C.SAMPLING.LANGEVIN_VAE = CN()
_C.SAMPLING.LANGEVIN_VAE.STEP_SIZE = 0.01
_C.SAMPLING.LANGEVIN_VAE.MOMENTUM = 0.
_C.SAMPLING.LANGEVIN_VAE.ITERATION = 10
_C.SAMPLING.LANGEVIN_VAE.DECAY_RATIO = 0.5
_C.SAMPLING.LANGEVIN_VAE.DECAY_STEP = 5

### dfo ###
_C.SAMPLING.DFO = CN()
_C.SAMPLING.DFO.RATIO = 0.1
_C.SAMPLING.DFO.ITERATION = 20
_C.SAMPLING.DFO.DECAY_RATIO = 0.5
_C.SAMPLING.DFO.DECAY_STEP = 10

### dmo ###
_C.SAMPLING.DMO = CN()
_C.SAMPLING.DMO.ITERATION = 5
_C.SAMPLING.DMO.THRESHOLD = -10.
_C.SAMPLING.DMO.LIMIT_SAMPLE = 16

###################
###### MODEL ######
###################

_C.MODEL = CN()

_C.MODEL.NAME = "Convnext-UNet" # Convnext-UNet, resnet18, vit_base_patch16_224
_C.MODEL.INOUT = "m" # Choice one among "m" "l" "l-m-l" "l-m". l-m-l means latent is decoded to motion and motion is fed into model. Then motion is encoded to latent.
_C.MODEL.TARGET = "auto"
_C.MODEL.CONV_DIMS = [96, 192, 384, 768]
_C.MODEL.ENC_LAYERS = ['convnext','convnext','convnext','convnext']
_C.MODEL.ENC_DEPTHS = [3,3,9,3]

_C.MODEL.DEC_LAYERS = ['convnext','convnext','convnext']
_C.MODEL.DEC_DEPTHS = [3,3,3]

_C.MODEL.EXTRACTOR_NAME = "query_uv_feature"
_C.MODEL.PREDICTOR_NAME = "Regressor_Transformer_with_cat_feature"

_C.MODEL.CONV_DROP_PATH_RATE = 0.1
_C.MODEL.ATTEN_DROPOUT_RATE = 0.1
_C.MODEL.QUERY_EMB_DIM = 128
_C.MODEL.NUM_ATTEN_BLOCK = 4

_C.VAE = CN()
_C.VAE.NAME = "VAE" # VAE, Transformer_VAE
_C.VAE.LATENT_NUM = 1
_C.VAE.LATENT_DIM = 32
_C.VAE.KLD_WEIGHT = 0.01


###############################
#### DIFFUSION for R2Diff #####
###############################

_C.DIFFUSION = CN()

_C.DIFFUSION.TYPE = "normal" # normal, improved, sde
_C.DIFFUSION.STEP = 1000
_C.DIFFUSION.MAX_RANK = 1 #  used for calculating end
_C.DIFFUSION.SIGMA = 1.0
_C.DIFFUSION.TARGET_MODE = "max"

# for normal DDPM
_C.DIFFUSION.START = 1e-5
_C.DIFFUSION.END = 2e-2

# for Improved DDPM
_C.DIFFUSION.S = 8e-3
_C.DIFFUSION.BIAS = 0.

# for classifier free guidance
_C.DIFFUSION.IMG_GUIDE = 1.0

# for retrieval-based evaluation
_C.DIFFUSION.STEP_EVAL = 500

###################
#### LATENT D ####
###################

_C.SAMR = CN()

_C.SAMR.STEP = 300

_C.SAMR.NOISE = CN()
_C.SAMR.NOISE.TYPE = "latent-gaussian" # gaussian, latent-gaussian, latent-range
_C.SAMR.NOISE.SCALE = "custom" # custom, auto

_C.SAMR.NOISE.CUSTOM = CN()
_C.SAMR.NOISE.CUSTOM.MAX = 5.0
_C.SAMR.NOISE.CUSTOM.MIN = 1e-2

####################
#### Denoise SM  ####
####################

_C.DSM = CN()
_C.DSM.STEP = 100

_C.DSM.NOISE = CN()
_C.DSM.NOISE.TYPE = "gaussian" # gaussian, latent_gaussian

_C.DSM.NOISE.GAUSSIAN = CN()
_C.DSM.NOISE.GAUSSIAN.START_STD = 1e-5
_C.DSM.NOISE.GAUSSIAN.END_STD = 2e-1

_C.DSM.NOISE.LATENT = CN()
_C.DSM.NOISE.LATENT.START_STD = 1e-2
_C.DSM.NOISE.LATENT.END_STD = 3.0

_C.DSM.INF = CN() #inf = inference
_C.DSM.INF.METHOD = "ALD" # ALD (Annealed Langevin dynamics)

_C.DSM.INF.T = 10
_C.DSM.INF.ETA = 0.1
_C.DSM.INF.STEP = 50
_C.DSM.INF.NOISE_WEIGHT = 0.001

###################
#### RETREIVE  ####
###################

_C.RETRIEVAL = CN()
_C.RETRIEVAL.RANK = 1

###################
###### OPTIM ######
###################

_C.OPTIM = CN()

_C.OPTIM.LR = 1e-4

_C.OPTIM.SCHEDULER = CN()
_C.OPTIM.SCHEDULER.GAMMA = 0.99
_C.OPTIM.SCHEDULER.STEP = 1000

###################
####### SDE #######
###################

_C.SDE = CN()

### SDE ###
_C.SDE.NAME = "vpsde" # "vpsde", "subvpsde", "vesde", "latent_vpsde", "irsde"
_C.SDE.N = 1000

# vpsdes
_C.SDE.VPSDES =CN()
_C.SDE.VPSDES.BETA_MIN = 1.
_C.SDE.VPSDES.BETA_MAX = 20.

# vesdes
_C.SDE.VESDES = CN()
_C.SDE.VESDES.SIGMA_MIN = 0.01
_C.SDE.VESDES.SIGMA_MAX = 90.

# irsdes
_C.SDE.IRSDES = CN()
_C.SDE.IRSDES.LAMDA = 0.5
_C.SDE.IRSDES.THETA = 2.0

### SDE TRAINING ### 
_C.SDE.TRAINING = CN()
_C.SDE.TRAINING.CONTINUOUS = True # `True` indicates that the score model was continuously trained.

### SDE SAMPLING ###
_C.SDE.SAMPLING = CN()
_C.SDE.SAMPLING.METHOD = "pc" # "pc" or "ode"
_C.SDE.SAMPLING.STEP = 0 # An integer. If this value over 0., the number of iteration is determined by this value.
_C.SDE.SAMPLING.NOISE_REMOVAL = True # If `True`, add one-step denoising to final samples.

# for pc sampling
_C.SDE.SAMPLING.PREDICTOR = "euler_maruyama" # euler_maruyama, reverse_diffusion, ancestral_sampling, cold_diffusion, none
_C.SDE.SAMPLING.CORRECTOR = "none" # none, langevin, ald
_C.SDE.SAMPLING.SNR = 0.16 # A `float` number. The signal-to-noise ratio for configuring correctors.
_C.SDE.SAMPLING.N_STEPS_EACH = 1 # An integer. The number of corrector steps per predictor update.
_C.SDE.SAMPLING.PROBABILITY_FLOW = True # If `True`, solve the reverse-time probability flow ODE when running the predictor. Check SDE reverse

#####################
####### Noise #######
#####################

_C.NOISE = CN()

_C.NOISE.NAME = "gaussian"

_C.NOISE.KNN = CN()
_C.NOISE.KNN.RANK = 3

