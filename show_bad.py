import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from visualize import label_image
from tqdm import tqdm
import numba
import sys


def norm_im(im):
    new_im = im.copy()
    for i in range(3):
        mean = np.mean(im[:, :, i])
        std = np.std(im[:, :, i])
        new_im[:, :, i] = (new_im[:,:,i]-mean)/std
    return new_im

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    roi = []
    visited = set()

    r_threshold = 250

    def explore(row, col, id, visited, roi, r_threshold):
        queue = [[row, col]]
        while queue != []:
            row, col = queue.pop(0)
            if row >= 0 and col >= 0 and row < len(I[:, 0, 0]) and col < len(I[0, :, 0]):
                if (row, col) not in visited:
                    visited.add((row, col))
                    if I[row, col, 0] >= r_threshold:
                        if row < roi[id][1]:
                            roi[id][1] = row
                        elif row > roi[id][3]:
                            roi[id][3] = row + 2
                        if col < roi[id][0]:
                            roi[id][0] = col
                        elif col > roi[id][2]:
                            roi[id][2] = col + 2
                        queue.append([row-1, col])
                        queue.append([row+1, col])
                        queue.append([row, col+1])
                        queue.append([row, col-1])

    id = 0
    for row in range(len(I[:, 0, 0])):
        for col in range(len(I[0, :, 0])):
            if (row, col) not in visited:
                visited.add((row, col))
                if I[row, col, 0] >= r_threshold:
                    roi.append([col, row, col+2, row+2])
                    explore(row - 1, col, id, visited, roi, r_threshold)
                    explore(row + 1, col, id, visited, roi, r_threshold)
                    explore(row, col + 1, id, visited, roi, r_threshold)
                    explore(row, col - 1, id, visited, roi, r_threshold)
                    id += 1

    def clipx(x):
        return max(0, min(len(I[0]), x))
    def clipy(y):
        return max(0, min(len(I), y))

    avg_im = Image.open("avg_red.png")
    avg_red = norm_im(np.asarray(avg_im).astype(np.float64))
    # Convolutions
    score_threshold = 350
    inflations = 4
    conv_range = 2
    for label in roi:
        break_flag = False
        dy = label[3] - label[1]
        y_orig = int((label[3] + label[1]) / 2)
        dx = label[2] - label[0]
        x_orig = int((label[2] + label[0]) / 2)
        d_orig = max(dx, dy)
        for xdiff in range(-conv_range, conv_range+1):
            x = clipx(x_orig + xdiff)
            for ydiff in range(-conv_range, conv_range+1):
                y = clipy(y_orig + ydiff)
                for inflation in range(inflations+1):
                    d = d_orig + inflation
                    new_im = I[clipy(y - d):clipy(y + d), clipx(x - d):clipx(x + d)]
                    im = Image.fromarray(new_im).resize((len(avg_red), len(avg_red[0])))
                    im = norm_im(np.asarray(im).astype(np.float64))
                    conv = im * avg_red
                    score = np.sum(conv)
                    if score > score_threshold:
                        bounding_boxes.append([clipx(x - d), clipy(y - d), clipx(x + d), clipy(y + d)])
                        break_flag = True
                        break
                if break_flag:
                    break
            if break_flag:
                break

    #bounding_boxes.append([tl_row, tl_col, br_row, br_col])

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'
#data_path = '../data/Labelled_reds'
label_path = './bad_images'
os.makedirs(label_path, exist_ok=True)  # create directory if needed

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# get sorted list of files: 
#file_names = sorted(os.listdir(data_path))
file_names = ["RL-036.jpg",
              "RL-037.jpg",
              "RL-025.jpg",
              "RL-022.jpg",
              "RL-021.jpg",
              "RL-016.jpg"]

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in tqdm(range(len(file_names))):
    # read image using PIL:
    im = Image.open(os.path.join(data_path, file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(im)
    
    preds[file_names[i]] = detect_red_light(I)
    # Saving images with bounding boxes
    labelled_im = label_image(im, preds[file_names[i]])
    labelled_im.save(os.path.join(label_path, file_names[i]))

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'bad_preds.json'),'w') as f:
    json.dump(preds,f)
with open('bad_preds.json','w') as f:
    json.dump(preds,f)
