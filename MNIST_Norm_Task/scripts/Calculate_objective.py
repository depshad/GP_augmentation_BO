import os
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import imutils
import cv2
import itertools
import argparse

def str2bool(v):
    print(v.lower())
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

my_parser = argparse.ArgumentParser(description='List the arguments')

my_parser.add_argument('--dataset_path',
                       type=str,
                       default=None)

my_parser.add_argument('--label_path',
                       type=str,
                       default=None)

my_parser.add_argument('--save_dir',
                       type=str,
                       default=None)

my_parser.add_argument('--sort_points',
                       type=str2bool,
                       default=False)



args = my_parser.parse_args()

dataset_path = args.dataset_path
label_path =   args.label_path

def black_box_func(X,ref):
    return -1*LA.norm(X-ref)

Y=np.load(label_path)

X = np.load(dataset_path)

X=X/255.
reference = X[3000]

y= np.array([np.round(black_box_func(img,ref=reference),4) for img in tqdm(X)])
print(X.shape,y.shape)

data_dir= args.save_dir

if args.sort_points:
    Z_train=np.array([])
    X_train,Y_train = X[:1000], y[:1000]
    X_val,Y_val= X[2000:], y[2000:]
    idx=np.argsort(-1*Y_train)
    X_train3,Y_train3 = X_train[idx[:100]],Y_train[idx[:100]]
    X_val,Y_val= X[2000:], y[2000:]
    np.savez(os.path.join(data_dir,'mnist_100.npz'), 
             X_train=X_train3,Y_train=Y_train3,X_val=X_val, 
             Y_val=Y_val,Z_train=Z_train,reference=reference)
    print(X_train3.shape,Y_train3.shape)
    print(np.max(Y_train3))
else:
    Z_train=np.array([])
    X_train,Y_train = X[:1000], y[:1000]
    X_val,Y_val= X[2000:], y[2000:]  
    idx = np.random.choice(X_train.shape[0], 100, replace=False)  
    X_train3,Y_train3 = X_train[idx[:100]],Y_train[idx[:100]]
    X_val,Y_val= X[2000:], y[2000:]
    np.savez(os.path.join(data_dir,'mnist_100.npz'), 
             X_train=X_train3,Y_train=Y_train3,X_val=X_val, 
             Y_val=Y_val,Z_train=Z_train,reference=reference)
    print(X_train3.shape,Y_train3.shape)
    print(np.max(Y_train3))

#Rotate an image
rotations = list(np.arange(0,360,20))[1:]
aug_rot_set = np.expand_dims(X_train3[0], axis=0)
label_rot = []
label_rot.append(Y_train3[0])
 
for i,img in enumerate(tqdm(X_train3)) :
    #save the y values
    y_rot = Y_train3[i]
    
    #performing all rotations for the image
    for rot in rotations :
        rot_img = imutils.rotate(img, angle=rot)
        rot_img = np.expand_dims(rot_img, axis=0)
        aug_rot_set = np.concatenate((aug_rot_set,rot_img), axis = 0)
        
        #appending labels
        label_rot.append(y_rot)

aug_rot_set = aug_rot_set[1:]
label_rot = label_rot[1:]

assert len(aug_rot_set) == len(label_rot)

#generate multiple translations in x and y directions
tx = np.arange(-7,7,1)
ty = np.arange(-7,7,1)
translations = list(itertools.product(tx,ty))

aug_trans_set = np.expand_dims(X_train3[0], axis=0)
label_trans = []
label_trans.append(Y_train3[0])

for i,img in enumerate(tqdm(X_train3)) :
    
    #getting height width of image
    height, width = img.shape[:2]
    
    #save the y values
    y_trans = Y_train3[i]
    
    #performing all rotations for the image
    for trans in translations :
        
        #getting the translation 
        T = np.float32([[1, 0, trans[0]], [0, 1, trans[1]]])
        trans_img = cv2.warpAffine(img, T, (width, height))
        trans_img = np.expand_dims(trans_img, axis=0)
        aug_trans_set = np.concatenate((aug_trans_set,trans_img), axis = 0)
        
        #appending labels
        label_trans.append(y_trans)

aug_trans_set = aug_trans_set[1:]
label_trans = label_trans[1:]

assert len(aug_trans_set) == len(label_trans)

#Smooth an image
smoothness = list(np.arange(0,2,0.05))[1:]
aug_smooth_set = np.expand_dims(X_train3[0], axis=0)
rand_noise = np.random.rand(784).reshape(28,28)

label_smooth = []
label_smooth.append(Y_train3[0])

for i,img in enumerate(tqdm(X_train3)) :
    
    #save the y values
    y_smooth = Y_train3[i]
    
    #performing all smoothness for the image
    for smooth in smoothness :
        eps = rand_noise * smooth
        smooth_img = img + eps
        smooth_img = np.expand_dims(smooth_img, axis=0)
        aug_smooth_set = np.concatenate((aug_smooth_set,smooth_img), axis = 0)
        
        #adding the y
        label_smooth.append(y_smooth)

aug_smooth_set = aug_smooth_set[1:]
label_smooth = label_smooth[1:]

assert len(aug_smooth_set) == len(label_smooth)
np.savez(os.path.join(data_dir,'augmented_images.npz'), 
         rot=aug_rot_set,
         rot_y = label_rot,
         trans=aug_trans_set,
         trans_y = label_trans,
         smooth=aug_smooth_set,
         smooth_y = label_smooth
        )