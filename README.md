# Uncertainty-aware Labelled Augmentations for High Dimensional Latent Space Bayesian Optimisation

## Abstract 

Black-box optimization problems are ubiquitous and of importance in many critical areas of science and engineering. Bayesian optimisation (BO) over the past
years has emerged as one of the most successful techniques for optimising expensive black-box objectives. However, efficient scaling of BO to high-dimensional
settings has proven to be extremely challenging. Traditional strategies based on projecting high-dimensional input data to a lower-dimensional manifold, such as
Variational autoencoders (VAE) and Generative adversarial networks (GAN) have improved BO performance in high-dimensional regimes, but their dependence on
excessive labeled input data has been widely reported. In this work, we target the data-greedy nature of deep generative models by constructing uncertainty-aware
task-specific labeled data augmentations using Gaussian processes (GPs). Our approach outperforms existing state-of-the-art methods on machine learning tasks
and demonstrates more informative data representation with limited supervision.

## Code

### Calculate_objective.py 

Create inital data points for bayesian optimisation. 

Set sort_points to True to get the highest valued initial data points to run subsequent optimisation.

#### Usage

 ```shell script

python ./MNIST_Norm_Task/scripts/Calculate_objective.py --dataset_path 'mnist_x.npy' \
                                                --label_path 'mnist_y.npy' \
                                                --save_dir 'path to save the initial data for BO' \
                                                --sort_points 'True'
 ```

### GP_augment_norm_task.py

#### Attributes
To run desired experiments, set appropriate flags (GP_aug_flag, naive_aug_flag, etc.) to True as the argument.

More implementation details can be found in the code itself.

result_dir : Directory to save results 

data_val_path : File path for validation data 

data_dir : Directory which contains the initial data points

vae_model : File path for pretrained vae model

#### Usage
 ```shell script

python ./MNIST_Norm_Task/scripts/GP_augment_norm_task.py --result_dir 'path to save results' \
                                                --data_val_path 'Created in Calculate_objective script' \
                                                --data_dir 'Directory path for inital data points' \
                                                --vae_model 'pretained vae model' \
                                                --naive_aug_flag 'True'

 ```
