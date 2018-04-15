# test2.py

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



# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # print('checking variables: ')
    # print(color_space)
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) 
        # plt.imshow(test_img)
        # plt.show()     
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #print(len(features))
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #print(prediction)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows




### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, gray
orient = 16  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 680] # Min and max in y to search in slide_window()


# load model
svc = pickle.load(open( "svc_ycrcb.pkl", "rb" ))
X_scaler = pickle.load(open( "X_scaler.pkl", "rb" ))



print("Testing...")
image = mpimg.imread('test_images/test6.jpg')
#draw_image = np.copy(image)
# convert draw image to desired color space:
#draw_image = convert_color(draw_image, conv='COLOR_RGB2YCrCb')
# draw_image = convert_color(draw_image, conv='COLOR_RGB2YCrCb')
# draw_image = convert_color(draw_image, conv='COLOR_RGB2HSV')

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255


ystart = 380
ystop = 680


# load model
svc = pickle.load(open( "svc_ycrcb.pkl", "rb" ))
X_scaler = pickle.load(open( "X_scaler.pkl", "rb" ))


def search_multiple_scales(img):

    draw_img = np.copy(img)
    # if the file type is .png, uncomment the following line
    # img = img.astype(np.float32)/255
    # img = image.astype(np.float32)/255
    img = convert_color(img, conv=color_space)
    # img = image.astype(np.float32)/255
    # define fixed parameters:
    ystart = 380
    ystop = 680
    overlap = 0.5
    # orient = 12  # HOG orientations
    # pix_per_cell = 12 # HOG pixels per cell
    # cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    # search the following window sizes:
    window_sizes = [48, 64, 92, 128] # [64, 128]
    y_start_stop = [[380, 500], [380, 580], [380, 680], [380, 680]]
    y_start_stop = [[380, 680], [380, 680], [380, 680], [380, 680]]

    # find windows to be tested:
    windows = []
    for i in range(len(window_sizes)):
        windows += slide_window(img, x_start_stop=[None, None], y_start_stop= [y_start_stop[i][0],y_start_stop[i][1]] , 
                        xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(overlap, overlap))

    #for i in range(len(window_sizes)):
     #   windows += slide_window(img, x_start_stop=[None, None], y_start_stop=[ystart, ystop], 
      #                  xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(overlap, overlap))

    #print(len(windows))
    #print("windows")
    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=(16, 16), hist_bins=16, 
                    hist_range=(0, 256), orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=False, 
                    hist_feat=False, hog_feat=True)
    #print("number of hot windows: ", len(hot_windows))
    # returns an image with labeled predictions

    # apply heat
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_windows)
    plt.imshow(heat)
    plt.show()

    # apply heat threshold
    heat = apply_threshold(heat,2)
    plt.imshow(heat)
    plt.show()

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 200)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #labels = label(windows)

    draw_img = draw_labeled_bboxes(draw_img, labels)
    plt.imshow(draw_img)
    plt.show()

    # window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 225), thick=6)                    

    return draw_img


window_img = search_multiple_scales(image)
#plt.imshow(window_img)
#plt.show()



'''
from moviepy.editor import VideoFileClip
test_result = 'test_video_labeled.mp4'
test_clip = VideoFileClip('test_video.mp4')
test_result_clip = test_clip.fl_image(find_cars_filtered)
test_result_clip.write_videofile(test_result, audio=False)
'''
'''
from moviepy.editor import VideoFileClip
full_result = 'full_video_labeled.mp4'
full_clip = VideoFileClip('project_video.mp4')
full_result_clip = full_clip.fl_image(search_multiple_scales)
full_result_clip.write_videofile(full_result, audio=False)
'''
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
