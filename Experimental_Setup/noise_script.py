#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import numpy as np
import glob
# Set parameters for Laplace function implementation
mean = 0
sensitivity = 255
folder_path = "./chest_xray" #Root Data Folder holding train/test folder 
# Take e = 0.25, 0.5 & 1
epsilon = 1
beta = sensitivity/epsilon #Amount of Spread
# Gets random laplacian noise for all values
print('hello')
for img_path in glob.iglob(folder_path +"/train/NORMAL/*.jpeg"):
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    Laplacian_noise = np.random.laplace(mean, beta, 1)
    noisy_image = np.zeros(img.shape, np.float32)
    noisy_image[:, :, :] = img[:, :, :]+ Laplacian_noise
    output_path = folder_path +"/train_%s/%s" % (epsilon, img_name)
    print(output_path)
    cv2.imwrite(output_path, noisy_image)
    break
print ("Completed eps = ", epsilon)


# In[ ]:




