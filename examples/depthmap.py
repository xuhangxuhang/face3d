import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
from glob import glob
import matplotlib.pyplot as plt
# sys.path.append('/workspace/Xu/FaceFitting-master/')

sys.path.append('..')
import face3d
from face3d import mesh_numpy, mesh
from face3d.morphable_model import MorphabelModel

import pandas as pd

import matplotlib.pyplot as pl

import cv2
import dlib

from tqdm import tqdm
import argparse
import configparser

def get_depth_image(image_path):
    im = cv2.imread(image_path)
    h, w, c = im.shape
    landmarks = pd.read_pickle(image_path.replace('.jpg','.pkl'))

    bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
    x = mesh.transform.from_image(landmarks, h, w)
    X_ind = bfm.kpt_ind


    fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter=200, isShow=False)
    colors = bfm.generate_colors(np.random.rand(bfm.n_tex_para, 1))
    colors = np.minimum(np.maximum(colors, 0), 1)

    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)


    triangles = bfm.triangles
    z = image_vertices[:,2:]
    z = z - np.min(z)
    z = z/np.max(z)
    attribute = z
    depth_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=1)
    depth_image = (depth_image*255).astype('uint8')

    savename = image_path.replace('.jpg','-depth.jpg')
    io.imsave(savename, np.squeeze(depth_image))     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='oulu file')
    parser.add_argument('-f',help='fold')
    arg = parser.parse_args()
    fold = int(arg.f)
    

    # split oulu dataset to 10 parts, accelerate the generation speed
    live_imgs = list(pd.read_csv('live_imgs.txt',header=None)[0].get_values())
    live_imgs = live_imgs[len(live_imgs)//2:]
    
    length = len(live_imgs)//10
    
    live_imgs = live_imgs[length*fold:length*(fold+1)]
    print('processing on fold {}'.format(fold))
    print('total imgs in fold {} is {}'.format(fold,len(live_imgs)))
    
    for img in tqdm(live_imgs):
        get_depth_image(img)
