import os, math, cv2, sys, time, torch, logging
import keyNet.config as config
import keyNet.aux.tools as aux
from train_utils import training_epochs, check_val_rep, fix_randseed

## Load data
from keyNet.datasets.pytorch_dataset import DatasetGeneration, pytorch_dataset
from torch.utils.data import DataLoader
## Network architecture & loss / optimizer
from keyNet.model.keynet_architecture import keynet
import torch.optim as optim
## Training loop
from tqdm import tqdm
import time
from keyNet.aux.logger import Logger

args = config.get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check directories
aux.check_directory('keyNet/data')
aux.check_directory(args.weights_dir)
aux.check_directory(args.synth_dir)

Logger.initialize(args)

# Set random seeds
fix_randseed(args.random_seed)

print('Start training Key.Net Architecture')

# Create Dataset
dataset_generation = DatasetGeneration(args.data_dir, args.synth_dir, args.patch_size, args.batch_size,
                                   args.max_angle, args.max_scale, args.max_shearing, args.random_seed, args.is_debugging, args.load_tfrecord)

training_data = dataset_generation.get_training_data()
validation_data = dataset_generation.get_validation_data()

dataset_train = pytorch_dataset(training_data, mode='train')
dataset_val = pytorch_dataset(validation_data, mode='val')

dataloader_train  = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_val  = DataLoader(dataset_val, batch_size=1, shuffle=False)


## network model configuration
# MSIP_sizes = [8, 16, 24, 32, 40]
# MSIP_factor_loss = [256.0, 64.0, 16.0, 4.0, 1.0]
MSIP_sizes = [int(i) for i in args.MSIP_sizes.split(",")]
MSIP_factor_loss =[float(i) for i in args.MSIP_factor_loss.split(",")]

print("MSIP hyperparameters : ", MSIP_sizes, MSIP_factor_loss)

model = keynet(args, device, MSIP_sizes)
model = model.to(device) ## use GPU

kernels = model.get_kernels(device) ## with GPU

if args.resume_training != '':
    model.load_state_dict(torch.load(args.resume_training))  ## Load the PyTorch learnable model parameters.
    logging.info("Model paramter : ", args.resume_training , " is loaded.")

## training configuration
epochs = args.num_epochs
epochs_val = args.epochs_val

learning_rate = args.init_initial_learning_rate  ## 0.5 after 20 epochs / 30 epochs converged.
decay_rate = args.learning_rate_decay_factor

## loss function and optimizer.
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1) ## weight decay (l2 regularizer) same as keynet paper.
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate) ## lr decay same as keynet paper.


## Count the number of learnable parameters.
logging.info("================ List of Learnable model parameters ================ ")
for n,p in model.named_parameters():
    if p.requires_grad:
        logging.info("{} {}".format(n, p.data.shape))
    else:
        logging.info("\n\n\n None learnable params {} {}".format( n ,p.data.shape))
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
logging.info("The number of learnable parameters : {} ".format(params.data))
logging.info("==================================================================== ")

## Run training 
best_rep_s = 0
best_epoch = 0
times_per_epoch = 0
with torch.no_grad():
    keynet_rep_val,_,_,_,_ = check_val_rep(dataloader_val, model, args.nms_size, device, num_points=25)
    best_rep_s = keynet_rep_val
    best_epoch = -1
    logging.info(('\n Epoch -1 : Repeatability Validation: {:.3f}.'.format(keynet_rep_val)))

## training loop
for epoch in range(epochs):
    training_epochs(epoch, dataloader_train, model, kernels, optimizer, MSIP_sizes, MSIP_factor_loss, args.weight_coordinates, args.patch_size, device)
    with torch.no_grad():
        rep_s, rep_m, error_overlap_s, error_overlap_m, possible_matches = check_val_rep(dataloader_val, model, args.nms_size, device, num_points=25)
    logging.info(('Epoch {} (Validation) : Repeatability (rep_s): {:.3f}. '.format(epoch, rep_s)))
    logging.info('\trep_m : {:.3f}, error_overlap_s : {:.3f}, error_overlap_m : {:.3f}, possible_matches : {:.3f}. \n'\
                .format( rep_m, error_overlap_s, error_overlap_m, possible_matches))

    if best_rep_s < rep_s:
        best_rep_s = rep_s
        best_epoch = epoch
        Logger.save_model(model, epoch, rep_s)

    if epochs == args.num_epochs_before_decay:
        ## Learning rate decay at epoch 20.
        scheduler.step()
    
print("Best validation repeatability score : {} at epoch {}. ".format(rep_s, best_epoch))