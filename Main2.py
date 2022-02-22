# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:10:26 2019

@author: asdg
"""
#coding:latin_1
import cv2
from Functions import *
from PIL import Image
from scipy.misc import imsave
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

resize_shape = (26,120)  # Resized image's shape
sigma = 1                # Noise standard dev.

window_shape = (10, 10)    # Patches' shape
step = 10                  # Patches' step
ratio = 1             # Ratio for the dictionary (training set).
ksvd_iter =5              # Number of iterations for the K-SVD.

#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. Image import. ----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

name = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/train1');
original_image = np.asarray(Image.open(name+'.png').convert('L').resize(resize_shape))
learning_image = np.asarray(Image.open('name_1.png').convert('L').resize(resize_shape))







'''img = cv2.imread('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/raw_photon_count.png')
# Here I want to convert img in 32 bits
cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB, img)
# Some image processing ...
cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR, img)
cv2.imwrite('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/raw_photon_count/out_32', img, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])

'''




#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
name1 = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/train1_27_07_2017');
noisy_image = np.asarray(Image.open(name1+'.png').convert('L').resize(resize_shape))
imsave(name + '2 - Noisy image.jpg', Image.fromarray(np.uint32(noisy_image)))
cmap='RdBu'
#-------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- 4. Denoising. -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

denoised_image, calc_time, n_total = denoising(noisy_image, learning_image, window_shape, step, sigma, ratio, ksvd_iter)

psnr = psnr(original_image, denoised_image)
print 'PSNR             : ' + str(psnr) + ' dB.'
imsave('resultant_image' + '.jpg', Image.fromarray(np.uint32(denoised_image)))
imsave(name + '3 - Out - Step ' + str(step) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(ratio) + '.jpg', Image.fromarray(np.uint32(denoised_image)))
imsave(name + '4 - Difference - Step ' + str(step) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(ratio) + '.jpg', Image.fromarray(np.uint32(np.abs(noisy_image - denoised_image))))

txt = open(name + ' Parameters.txt', 'a')
txt.write('INITIAL SETTINGS\n----------------' +
              '\n\nResizing shape : ' + str(resize_shape) +
              '\nWindow shape   : ' + str(window_shape) +
              '\nWindow step    : ' + str(step) +
              '\nResizing shape : ' + str(resize_shape) +
              '\nSigma          : ' + str(sigma) +
              '\nLearning ratio : ' + str(ratio) +
              '\nK-SVD iter.    : ' + str(ksvd_iter) +
              '\n\nComputation time : ' + str(calc_time) + ' seconds.' +
              '\nPSNR           : ' + str(psnr) + ' dB.\n')
txt.close()


plt.figure(1)
fig = plt.figure(figsize=(9,8));
plt.imshow(noisy_image, cmap='RdBu')
plt.imshow(denoised_image, cmap='RdBu')
#plt.imshow(noisy_image-denoised_image, cmap='RdBu')
plt.xlabel('_')
plt.ylabel('_')
#plt.colorbar(label='photon ')
plt.title('Denoised data')
#plt.colorbar(ticks=range(20000), label='photon count')
#plt.clim(1,10)









