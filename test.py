import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
import pickle
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label




### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 650] # Min and max in y to search in slide_window()


# load model
svc = pickle.load(open( "svc_ycrcb.pkl", "rb" ))
X_scaler = pickle.load(open( "X_scaler.pkl", "rb" ))



print("Testing...")
image = mpimg.imread('test_images/test5.jpg')
draw_image = np.copy(image)
# convert draw image to desired color space:
draw_image = convert_color(draw_image, conv='RGB2YCrCb')

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255


ystart = 400
ystop = 650
scale = 1.5
# heatmap = threshold(heatmap, 2)
# labels = label(heatmap)

out_img, bbox_list = find_cars(draw_image, ystart,color_space, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
plt.imshow(out_img)
plt.show()

# Add heat to each box in box list
heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat,bbox_list)
plt.imshow(heat)
plt.show()
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)
plt.imshow(heat)
plt.show()
# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 100)
# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)
plt.imshow(draw_img)
plt.show()

'''
from moviepy.editor import VideoFileClip
test_result = 'test_video_labeled.mp4'
test_clip = VideoFileClip('test_video.mp4')
test_result_clip = test_clip.fl_image(find_cars_filtered)
test_result_clip.write_videofile(test_result, audio=False)
'''

from moviepy.editor import VideoFileClip
full_result = 'full_video_labeled.mp4'
full_clip = VideoFileClip('project_video.mp4')
full_result_clip = full_clip.fl_image(find_cars_filtered)
full_result_clip.write_videofile(full_result, audio=False)

'''
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)       
'''             



