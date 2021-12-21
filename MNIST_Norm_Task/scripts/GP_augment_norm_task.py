# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision.utils import save_image
import torch.optim as optim
from typing import Optional, Union
from botorch.acquisition.analytic import ExpectedImprovement
import gpytorch
from tqdm import tqdm
from numpy import linalg as LA
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
#
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
torch.manual_seed(1)
np.random.seed(2)
from pytorch_metric_learning import distances
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
def str2bool(v):
    print(v.lower())
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

my_parser = argparse.ArgumentParser(description='List the arguments')

my_parser.add_argument('--GP_aug_flag',
                       type=str2bool,
                       default=False)
my_parser.add_argument('--LCB_flag',
                       type=str2bool,
                       default=False)
my_parser.add_argument('--vae_uncertain_flag',
                       type=str2bool,
                       default=False)                       
my_parser.add_argument('--triplet_loss_flag',
                       type=str2bool,
                       default=False)
my_parser.add_argument('--contrastive_loss_flag',
                       type=str2bool,
                       default=False)
my_parser.add_argument('--baseline_flag',
                       type=str2bool,
                       default=False)
my_parser.add_argument('--naive_aug_flag',
                       type=str2bool,
                       default=False)    

my_parser.add_argument('--naive_aug_rot_flag',
                       type=str2bool,
                       default=False)   
my_parser.add_argument('--naive_aug_transl_flag',
                       type=str2bool,
                       default=False)   
my_parser.add_argument('--naive_aug_smooth_flag',
                       type=str2bool,
                       default=False)   

args = my_parser.parse_args()


GP_aug_flag=args.GP_aug_flag
LCB_flag = args.LCB_flag
vae_uncertain_flag = args.vae_uncertain_flag

triplet_loss_flag= args.triplet_loss_flag
contrastive_loss_flag = args.contrastive_loss_flag

baseline_flag=args.baseline_flag
naive_aug_flag = args.naive_aug_flag
naive_aug_rot_flag=args.naive_aug_rot_flag
naive_aug_transl_flag=args.naive_aug_transl_flag
naive_aug_smooth_flag=args.naive_aug_smooth_flag



# Create a new directory for the new data points
my_parser.add_argument('--result_dir',
                        type=str,
                       default=None)   
root_dir=args.result_dir
my_parser.add_argument('--data_val_path',
                       type=str,
                       default=None)  
data_val_path=args.data_val_path
data_val = np.load(data_val_path)
my_parser.add_argument('--data_dir',
                       type=str,
                       default=None)   
data_dir=args.data_dir

my_parser.add_argument('--vae_model',
                       type=str,
                       default=None)   

vae_init_model_path = args.vae_model


NUM_DATA=100
true_data_path1=os.path.join(data_dir,'mnist_'+str(NUM_DATA)+'.npz')

if (triplet_loss_flag == contrastive_loss_flag ) and triplet_loss_flag == True:
    raise AssertionError("Triplet and Contrastive should not be both True")

class ContrastiveLossTorch:

    def __init__(self, threshold: float, hard: Optional[bool] = None):
        self.threshold = threshold
        self.hard = hard if hard is not None else False

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        loss = torch.zeros_like(emb_distance_matrix).to(embs)

        threshold_matrix = self.threshold * torch.ones(loss.shape).to(embs)

        high_dy_filter = y_distance_matrix > self.threshold
        aux_max_dz_thr = torch.maximum(emb_distance_matrix, threshold_matrix)
        aux_min_dz_thr = torch.minimum(emb_distance_matrix, threshold_matrix)

        if self.hard:
            # dy - dz
            loss[high_dy_filter] = y_distance_matrix[high_dy_filter] - emb_distance_matrix[high_dy_filter]
            # dz
            loss[~high_dy_filter] = emb_distance_matrix[~high_dy_filter]
        else:
            # (2 - min(threshold, dz) / threshold) * (dy - max(dz, threshold))
            loss[high_dy_filter] = (2 - aux_min_dz_thr[high_dy_filter]).div(self.threshold) * (
                    y_distance_matrix[high_dy_filter] - aux_max_dz_thr[high_dy_filter])

            #  max(threshold, dz) / threshold * (min(dz, threshold) - dy)
            loss[~high_dy_filter] = aux_max_dz_thr[~high_dy_filter].div(self.threshold) * (
                    aux_min_dz_thr[~high_dy_filter] - y_distance_matrix[~high_dy_filter])

        loss = torch.relu(loss)
        return loss

    def compute_loss(self, embs: Tensor, ys: Tensor):
        loss_matrix = torch.triu(self.build_loss_matrix(embs, ys), diagonal=1)
        n = (loss_matrix > 0).sum()

        if n == 0:
            n = 1
        # average over non-zero elements
        # return loss_matrix.sum().div(n)
        return loss_matrix.sum()

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, hard: Optional[bool] = None) -> str:
        metric_id = f'contrast-thr-{threshold:g}'
        if hard:
            metric_id += '-hard'
        return metric_id



class TripletLossTorch:
    def __init__(self, threshold: float, margin: Optional[float] = None, soft: Optional[bool] = False,
                 eta: Optional[float] = None):
        """
        Compute Triplet loss
        Args:
            threshold: separate positive and negative elements in temrs of `y` distance
            margin: hard triplet loss parameter
            soft: whether to use sigmoid version of triplet loss
            eta: parameter of hyperbolic function softening transition between positive and negative classes
        """
        self.threshold = threshold
        self.margin = margin
        self.soft = soft
        assert eta is None or eta > 0, eta
        self.eta = eta

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        positive_embs = emb_distance_matrix.where(y_distance_matrix <= self.threshold, torch.tensor(0.).to(embs))
        negative_embs = emb_distance_matrix.where(y_distance_matrix > self.threshold, torch.tensor(0.).to(embs))

        loss_loop = 0 * torch.tensor([0.], requires_grad=True).to(embs)
        n_positive_triplets = 0
        for i in range(embs.size(0)):
            pos_i = positive_embs[i][positive_embs[i] > 0]
            neg_i = negative_embs[i][negative_embs[i] > 0]
            pairs = torch.cartesian_prod(pos_i, -neg_i)
            if self.soft:
                triplet_losses_for_anchor_i = torch.nn.functional.softplus(pairs.sum(dim=-1))
                if self.eta is not None:
                    # get the corresponding delta ys
                    pos_y_i = y_distance_matrix[i][positive_embs[i] > 0]
                    neg_y_i = y_distance_matrix[i][negative_embs[i] > 0]
                    pairs_y = torch.cartesian_prod(pos_y_i, neg_y_i)
                    assert pairs.shape == pairs_y.shape, (pairs_y.shape, pairs.shape)
                    triplet_losses_for_anchor_i = triplet_losses_for_anchor_i * \
                                                  self.smooth_indicator(self.threshold - pairs_y[:, 0]) \
                                                      .div(self.smooth_indicator(self.threshold)) \
                                                  * self.smooth_indicator(pairs_y[:, 1] - self.threshold) \
                                                      .div(self.smooth_indicator(1 - self.threshold))
            else:
                triplet_losses_for_anchor_i = torch.relu(self.margin + pairs.sum(dim=-1))
            n_positive_triplets += (triplet_losses_for_anchor_i > 0).sum()
            loss_loop += triplet_losses_for_anchor_i.sum()
        # loss_loop = loss_loop.div(max(1, n_positive_triplets))

        return loss_loop

    def smooth_indicator(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, float):
            return np.tanh(x / (2 * self.eta))
        return torch.tanh(x / (2 * self.eta))

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, margin: Optional[float] = None, soft: Optional[bool] = None,
                      eta: Optional[bool] = None) -> str:
        if margin is not None:
            return f'triplet-thr-{threshold:g}-mrg-{margin:g}'
        if soft is not None:
            metric_id = f'triplet-thr-{threshold:g}-soft'
            if eta is not None:
                metric_id += f'-eta-{eta:g}'
            return metric_id

def score_image_recognition(X,ref):
    
    #current shape is bs*1*28*28 ---> reshape : bs * 784
    X1 = X.view(X.shape[0],784)
    ref = torch.as_tensor(ref)
    ref1 = ref.repeat(X.shape[0],1)
    
    return -1* torch.norm(X1-ref1, dim=1)

"""#### Training VAE-Simplistic Model"""

class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)   ### mean and sigma dimension 20

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std ## sampling from q(z)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) #encoder mu and log var of q(z)
        z = self.reparameterize(mu, logvar) # sample from q(z)
        return self.decode(z), mu, logvar #decode this

def encode(input_data):
    input_data = input_data.to(dtype = dtype, device = device)
    with torch.no_grad():
        mu,logvar = vae_model.encode(input_data.view(-1, 784))
        z = vae_model.reparameterize(mu, logvar) # sample from q(z)
    return z, mu,logvar

def decode(input_z):
    input_z = input_z.to(dtype = dtype, device = device)
    with torch.no_grad():
        decoded = vae_model.decode(input_z)
    return decoded.view(input_z.shape[0], 1, 28, 28)

def rank_weights(properties: np.array, k_val: float):
    """
    Calculates rank weights assuming maximization.
    Weights are not normalized.
    """
    if np.isinf(k_val):
        return np.ones_like(properties)
    
    ranks = np.argsort(np.argsort( -1 * properties))
    weights = 1.0 / (k_val * len(properties) + ranks)
    return weights

# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(recon_x, x, mu, logvar,y):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if triplet_loss_flag :
        triplet_loss = TripletLossTorch(
            threshold=0.1,
            margin=None,
            soft=True,
            eta=None
        )
        z_sample,_,_=encode(x)
        y=y.to(device)
        metric_loss = triplet_loss(z_sample, y)
        return BCE + KLD + metric_loss, BCE, KLD, metric_loss

    elif contrastive_loss_flag:
        constr_loss = ContrastiveLossTorch(threshold=0.1,
                                                   hard=True)
        z_sample,_,_=encode(x)
        metric_loss = constr_loss(z_sample, y)
        return BCE + KLD + 2*metric_loss, BCE, KLD, metric_loss
        
    else:
        return BCE + KLD, BCE, KLD

#Defines the training loop for one epoch for the entire batch

def train(epoch,train_loader):
    
    #initiliaze
    vae_model.train()
    train_loss = 0
    
    print ('Running Train DataLoader')
    
    for batch_idx, (data,y_train) in enumerate(train_loader):
        
        data = data.to(dtype = dtype, device = device)
        optimizer.zero_grad()
        #return from the model
        recon_batch, mu, logvar = vae_model(data)
        #computing the loss
        if triplet_loss_flag:
            loss,BCE,KD,triplet_loss = loss_function(recon_batch, data, mu, logvar,y_train)
        elif contrastive_loss_flag:
            loss,BCE,KD,contra_loss = loss_function(recon_batch, data, mu, logvar,y_train)
        else:
            loss,BCE,KD = loss_function(recon_batch, data, mu, logvar,y_train)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
    #printing and returning output
    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
    if triplet_loss_flag:
        print(f' BCE : {BCE}, KD : {KD}, triplet : {triplet_loss}')
    if contrastive_loss_flag:
        print(f' BCE : {BCE}, KD : {KD}, contra : {contra_loss}')
    
    return train_loss / len(train_loader.dataset)

#Defines the testing loop for one epoch for the entire batch

def test(epoch,test_loader):
    
    #model in eval mode
    vae_model.eval()
    test_loss = 0
    
    print ('Running Test DataLoader')
    with torch.no_grad():
        
        #returning the loss vals
        for batch_idx, (data,y_val) in enumerate(test_loader):
            data = data.to(dtype = dtype, device = device)
            recon_batch, mu, logvar = vae_model(data)
            if triplet_loss_flag:
                loss,BCE,KD,triplet_loss = loss_function(recon_batch, data, mu, logvar,y_val)
            elif contrastive_loss_flag:
                loss,BCE,KD,contra_loss = loss_function(recon_batch, data, mu, logvar,y_val)
            else:
                loss,BCE,KD = loss_function(recon_batch, data, mu, logvar,y_val)
            test_loss += loss.item()
            
    #printing and returning output
    print('====> Epoch: {} Average Val loss: {:.4f}'.format(
              epoch, test_loss / len(test_loader.dataset)))
        
    return test_loss / len(test_loader.dataset)

#Convert to Torch tensors
def get_tensor_dataset(data):
    data = torch.as_tensor(data, dtype=torch.float32)
    data = torch.unsqueeze(data, 1)
    return data



d = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fitted_model(train_latent, train_label, state_dict=None):
    
    # initialize and fit model
    model = SingleTaskGP(train_X=train_latent, train_Y=train_label).to(device)
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
    mll.to(train_latent)
    fit_gpytorch_model(mll)
    return model



def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    
    # optimize
    candidates,_  = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(d, device=device), 
            torch.ones(d, device=device),
        ]),
        q= n_ret_bo,
        num_restarts=10,
        raw_samples=256,
    )

    # observe new values 
    new_z = unnormalize(candidates.detach(), bounds=bounds)
    new_x = decode(new_z)
    new_obj = score_image_recognition(decode(new_z),ref).unsqueeze(-1)  # add output dimension
    return new_z, new_x, new_obj

"""### Data Augmentation

+ Create Fixed Augmentation for Static Images in the Training Dataset
+ Rotate, Translate,Gaussian Blur
+ Predict from GP and Add to the Dataset
"""

#dataset:  load the dataset everytime for training
data = np.load(true_data_path1)
train_data = data['X_train']
train_lab =  data['Y_train']
reference = data['reference']
ref = reference.reshape(-1)
ref=torch.from_numpy(ref).to(device)

train_data.shape

augment_data_path=os.path.join(data_dir,'augmented_images_'+ str(NUM_DATA)+'.npz') 
augment_data = np.load(augment_data_path)
aug_rot_set = augment_data['rot']
aug_trans_set =  augment_data['trans']
aug_smooth_set = augment_data['smooth']

#getting the labels to be used only for naive augmentation
aug_rot_y =  augment_data['rot_y']
aug_trans_y = augment_data['trans_y']
aug_smooth_y = augment_data['smooth_y']

"""#### Get the total augmented dta"""

total_augment = np.concatenate((aug_rot_set,aug_trans_set,aug_smooth_set), axis=0)

total_augment.shape

def flag_string(): 
  flag_string=''
  if triplet_loss_flag:
      temp=f'triplet_loss_flag='.split('=')[0]
      flag_string += '_' +temp
  if contrastive_loss_flag:
      temp=f'contrastive_loss_flag='.split('=')[0]
      flag_string += '_' +temp
  if baseline_flag:
      temp=f'baseline_flag='.split('=')[0]
      flag_string += '_' +temp
  if GP_aug_flag:
      temp=f'GP_aug_flag='.split('=')[0]
      flag_string += '_' +temp
  if vae_uncertain_flag:
      temp=f'vae_uncertain_flag='.split('=')[0]
      flag_string += '_' +temp
  if LCB_flag:
      temp=f'LCB_flag='.split('=')[0]
      flag_string += '_' +temp
  if naive_aug_flag:
      temp=f'naive_aug_flag='.split('=')[0]
      flag_string += '_' +temp
  if naive_aug_rot_flag:
      temp=f'naive_aug_rot_flag='.split('=')[0]
      flag_string += '_' +temp
  if naive_aug_transl_flag:
      temp=f'naive_aug_transl_flag='.split('=')[0]
      flag_string += '_' +temp
  if naive_aug_smooth_flag:
      temp=f'naive_aug_smooth_flag='.split('=')[0]
      flag_string += '_' +temp  
  return flag_string

from pathlib import Path
retraining_freq = 20
epochs = 30
gp_state_dict = None
bo_iter = 20
n_ret_bo = 3
MC_SAMPLES = 2048
d = 20

flags = flag_string()
print(f'Experiment :: {flags}')

vae_model_sv_path_root = os.path.join(root_dir,'vae_model_checkpoints',flags)
Path(vae_model_sv_path_root).mkdir(parents=True, exist_ok=True)

vae_loss_path = os.path.join(root_dir,'vae_loss_path',flags)
Path(vae_loss_path).mkdir(parents=True, exist_ok=True)

decode_img_path = os.path.join(root_dir,'decode_img_path',flags)
Path(decode_img_path).mkdir(parents=True, exist_ok=True)

gp_augment_path = os.path.join(root_dir,'GP_Augment_data',flags)
Path(gp_augment_path).mkdir(parents=True, exist_ok=True)

gp_model_path=os.path.join(root_dir,'GP_Model',flags)
Path(gp_model_path).mkdir(parents=True, exist_ok=True)

gp_data_path=os.path.join(root_dir,'GP_Data',flags)
Path(gp_data_path).mkdir(parents=True, exist_ok=True)

bounds = torch.tensor([[-6.0] * d, [6.0] * d], device=device, dtype=dtype)
k_val = 1e-3

from shutil import copyfile
# Copy data to root directory with flag names appended to the file
true_data_path = os.path.join(root_dir,flags)+ '.npz'
data_path=os.path.join(root_dir,'data_og.npz')
copyfile(true_data_path1, true_data_path)
copyfile(true_data_path1, data_path)


#loading
x_val = data_val['X_val']
y_val = data_val['Y_val']

#converting to tensor
val_x = get_tensor_dataset(x_val).float()
y_val_tensor = torch.as_tensor(np.array(y_val).reshape(-1,1))
m, M = y_val_tensor.min(), y_val_tensor.max()
if m == M:
    y_val_tensor= np.ones_like(y_val_tensor)

else:
    y_val_tensor= (y_val_tensor - m) / (M - m)

val_dataset = TensorDataset(val_x, y_val_tensor)

#defining the sampler for train
val_weights = rank_weights(y_val,k_val)
val_sampler = WeightedRandomSampler(val_weights, num_samples=len(val_weights), replacement=True)

"""### Total Training Loop - Weighted Retraining with GP Augment"""

print ('Starting the Process ')


for vae_step in tqdm(range(retraining_freq)):
    
    print (vae_step)
    
    
    #dataset load the dataset evrytime load the dataset for the training
    data = np.load(true_data_path)

    train_x = data['X_train']
    train_y = data['Y_train']
    train_z = data['Z_train']

    #Preparing the data
    x_train_tensor = get_tensor_dataset(train_x)
    y_train_tensor = torch.as_tensor(np.array(train_y).reshape(-1,1))

    z_train_tensor = torch.as_tensor(train_z)
   

    #Loading the model and saving
    vae_model = VAE().to(dtype=dtype, device=device)
    
    if not baseline_flag :
        #after first iteration
        if vae_step == 0:
            vae_state_dict = torch.load(vae_init_model_path,map_location=device)
            vae_model.load_state_dict(vae_state_dict)
        
        elif vae_step > 0 : 
            vae_state_dict = torch.load(vae_model_sv_path,map_location=device)
            vae_model.load_state_dict(vae_state_dict)
    else:
        pass
        
    #Defining optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)


    ############### Perform Augmentation #################################
    
    print ('Adding Augmented Data to VAE')
    
    #Adding Augmented Data
    if GP_aug_flag:

        if vae_step > 0 :
            #Converting to torch tensor
            x_gen = get_tensor_dataset(total_augment)
            x_gen=x_gen.to(device)
            #Encoding Augmented Data
            aug_z, mu_z, logvar_z  = encode(x_gen)

            if vae_uncertain_flag:
            
                #Uncertainty management
                var_z = torch.exp(logvar_z)
                var_sum = var_z.sum(axis=1)
                
                #scaled sum
                var_sum_sc = (var_sum - 20)
                idx_vae_ui = torch.where(var_sum_sc <= 0)[0]

                #get new aug_Z
                aug_z = torch.index_select(aug_z, 0, idx_vae_ui)
                
                #update the aug_X
                x_gen = torch.index_select(x_gen, 0, idx_vae_ui)
        
            # Getting the GP predictions
            f_preds = gp_model(normalize(aug_z, bounds=bounds))
            f_mean = f_preds.mean
            f_var =  f_preds.variance
            
            #mean and sigma of y
            obj_mu = train_y.mean().repeat(x_gen.shape[0])
            obj_std = train_y.std().repeat(x_gen.shape[0])
            
            #Converting to original scale
            y_gen = obj_mu + obj_std * f_mean.detach().cpu().numpy()
            if LCB_flag:
                y_lcb = obj_mu + obj_std * (f_mean-f_var).detach().cpu().numpy()
            else:
                y_lcb = obj_mu + obj_std * f_mean.detach().cpu().numpy()

            #Want to Sort and Take only Meaningful x taking top 3000
            idx_max = np.argsort(y_lcb)[::-1][0:3000]
            y_aug = y_gen[idx_max]
            x_aug = torch.cat([x_gen[i].unsqueeze(0) for i in idx_max] , dim=0).float()
            
            #update training points
            x_train_tensor = x_train_tensor.to(device)
            x_train_tensor =  torch.cat((x_train_tensor, x_aug))
            train_y = np.concatenate((train_y,y_aug), axis=0)
            
            #save the augmented data going into VAE using GP augmentations
            np.savez_compressed(
                gp_augment_path +'/' + str(vae_step) + '.npz',
                Y_new = y_aug,
                X_new = x_aug.cpu().numpy()
            )
            
    #Naive augmentation
    if naive_aug_flag:
        # np.random.seed(0)
        #choose ids
        idx_rot = np.random.choice(aug_rot_set.shape[0], 1000, replace=False)  
        idx_trans = np.random.choice(aug_trans_set.shape[0], 1000, replace=False)  
        idx_smooth = np.random.choice(aug_smooth_set.shape[0], 1000, replace=False)  
        
        #subset data
        aug_rot_set_nv = aug_rot_set[idx_rot]
        aug_rot_y_nv = aug_rot_y[idx_rot]
        aug_trans_set_nv = aug_trans_set[idx_trans]
        aug_trans_y_nv = aug_trans_y[idx_rot]
        aug_smooth_set_nv = aug_smooth_set[idx_smooth]
        aug_smooth_y_nv = aug_smooth_y[idx_rot]
        
        #combine
        total_augment_nv = np.concatenate((aug_rot_set_nv,aug_trans_set_nv,aug_smooth_set_nv), axis=0)
        
        #total y
        y_aug_nv = np.concatenate((aug_rot_y_nv,aug_trans_y_nv,aug_smooth_y_nv), axis=0)
        
        #get
        x_aug_nv = get_tensor_dataset(total_augment_nv)
    
        #update training points
        x_train_tensor =  torch.cat((x_train_tensor, x_aug_nv))
        train_y = np.concatenate((train_y,y_aug_nv), axis=0)
    
    if naive_aug_rot_flag:
        # np.random.seed(0)
        idx_rot = np.random.choice(aug_rot_set.shape[0], 3000, replace=False)  
        aug_rot_set_nv = aug_rot_set[idx_rot]
        aug_rot_y_nv = aug_rot_y[idx_rot]

        x_aug_nv = get_tensor_dataset(aug_rot_set_nv)
        y_aug_nv = aug_rot_y_nv

        x_train_tensor =  torch.cat((x_train_tensor, x_aug_nv))
        train_y = np.concatenate((train_y,y_aug_nv), axis=0)

    if naive_aug_transl_flag:
        # np.random.seed(0)
        idx_trans = np.random.choice(aug_trans_set.shape[0], 3000, replace=False) 
        aug_trans_set_nv = aug_trans_set[idx_trans]
        aug_trans_y_nv = aug_trans_y[idx_trans]  

        x_aug_nv = get_tensor_dataset(aug_trans_set_nv)
        y_aug_nv = aug_trans_y_nv
        
        x_train_tensor =  torch.cat((x_train_tensor, x_aug_nv))
        train_y = np.concatenate((train_y,y_aug_nv), axis=0)
    
    if naive_aug_smooth_flag:
        # np.random.seed(0)
        idx_smooth = np.random.choice(aug_smooth_set.shape[0], 3000, replace=False)  
        aug_smooth_set_nv = aug_smooth_set[idx_smooth]
        aug_smooth_y_nv = aug_smooth_y[idx_smooth]

        x_aug_nv=get_tensor_dataset(aug_smooth_set_nv)
        y_aug_nv=aug_smooth_y_nv

        x_train_tensor =  torch.cat((x_train_tensor, x_aug_nv))
        train_y = np.concatenate((train_y,y_aug_nv), axis=0)
    
    y_train_tensor = torch.as_tensor(np.array(train_y).reshape(-1,1))

    if triplet_loss_flag or contrastive_loss_flag:
      m, M = y_train_tensor.min(), y_train_tensor.max()
      if m == M:
          y_train_tensor= np.ones_like(y_train_tensor)
      else:
          y_train_tensor = (y_train_tensor - m) / (M - m)

    #create tensor dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
    #defining the sampler for train
    train_weights = rank_weights(train_y,k_val) 
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    
    #Converting to Dataloaders
    train_loader =  DataLoader(train_dataset, batch_size=32,num_workers=0,sampler= train_sampler,drop_last=True)
    
    #Converting to Dataloaders
    val_loader =  DataLoader(val_dataset, batch_size=32,num_workers=0,sampler= val_sampler,drop_last=True)

    #saving loss per epochs
    train_loss = []
    val_loss = []
    
    #Number of Training epochs
    for epoch in range(1, epochs + 1):
        
        print (epoch)
        
        #training step
        loss1 = train(epoch,train_loader)
        train_loss.append(loss1)
        
        
        #validation step
        test_loss = test(epoch,val_loader)
        val_loss.append(test_loss)
        
        
        #saving latent images to understand the vae prediction. For now same folder will save all iterations VAE 
        with torch.no_grad():
            sample = torch.randn(64, 20).to(dtype = dtype, device=device)
            sample = vae_model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       decode_img_path +'/' +'sample_' + str(epoch) +'_' + str(vae_step) + '.png')


    #saving the loss functions at files after completing the epochs
    
    np.savez_compressed(vae_loss_path + '/' +str(vae_step) + '.npz',
            loss_train = train_loss,
            loss_test =  val_loss)
    
    #saving the model after epochs
    vae_model_sv_path = vae_model_sv_path_root + '/vae_iter_'+ str(vae_step) + '.pt'
    torch.save(vae_model.state_dict(), vae_model_sv_path)
    
    print ('VAE part done for step_' + str(vae_step))
   
    ###############ends for vae ########################
    
    ########################## Bayesian Opt ########################################################
    
    #Reading the data for GP
    #dataset load the dataset evrytime load the dataset for the training
    data = np.load(true_data_path)
    train_x_gp = data['X_train']
    train_y_gp = data['Y_train']

    #Preparing the data
    x_train_tensor_gp = get_tensor_dataset(train_x_gp).to(device)
    y_train_tensor_gp = torch.as_tensor(np.array(train_y_gp).reshape(-1,1)).float().to(device)
    z_train_tensor_gp,_,_  = encode(x_train_tensor_gp) ##### getting encoding from curr model

    #tracking the max value with which
    best_value = y_train_tensor_gp.max().item()
    
    print(f"\nRunning BO ", end='')
    best_observed = []

    #run N_BATCH rounds of BayesOpt after the initial random batch
    
    for iteration in tqdm(range(bo_iter)):   
        
        print ('Starting Iteration for step_' + str(iteration))
        
        #fit the GP model with initial number of points
        gp_model = get_fitted_model(
            normalize(z_train_tensor_gp, 
                      bounds=bounds), 
            standardize(y_train_tensor_gp), 
            state_dict= gp_state_dict)
            
        # Save the training data for GP
        np.savez_compressed(
            gp_data_path+'/' + 'GP_'+str(vae_step)+'_'+str(iteration),
            Z_train= z_train_tensor_gp.cpu().numpy(),
            Y_train= y_train_tensor_gp.cpu().numpy().squeeze(1),
            X_train = x_train_tensor_gp.cpu().squeeze(1).numpy()
            
        )
        # define the qNEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
        qEI = qExpectedImprovement(model= gp_model, sampler=qmc_sampler,best_f=standardize(y_train_tensor_gp).max())
    

        #optimize and get new observation
        new_z, new_x, new_obj = optimize_acqf_and_get_observation(qEI) # gives the new_X
                
        #update training points
        x_train_tensor_gp = torch.cat((x_train_tensor_gp, new_x))
        z_train_tensor_gp = torch.cat((z_train_tensor_gp, new_z))
        y_train_tensor_gp = torch.cat((y_train_tensor_gp, new_obj))
                
        # update progress
        best_value = y_train_tensor_gp.max().item()
        best_observed.append(best_value)

        gp_state_dict = gp_model.state_dict()
       
                
        torch.save(gp_model.state_dict(), gp_model_path +'/' + 'GP_'+str(vae_step)+'_'+str(iteration)+'.pth')
        
    #Saving the new data
    np.savez_compressed(
            true_data_path,
            Z_train= z_train_tensor_gp.cpu().numpy(),
            Y_train= y_train_tensor_gp.cpu().numpy().squeeze(1),
            X_train = x_train_tensor_gp.cpu().squeeze(1).numpy()
            
        )



