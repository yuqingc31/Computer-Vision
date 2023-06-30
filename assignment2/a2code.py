import numpy as np
import cv2

# Function
# load the location of the image and change it to grey scale image 
def findIndex(start_index, end_index, path):
    images = []
    for i in range(start_index, end_index):
        cur_idx = str(i).zfill(3)
        img_grey = cv2.imread(path + cur_idx + '.jpg', cv2.IMREAD_GRAYSCALE)
        images.append(img_grey)

    return images

def compute_descriptors(img, features = None):
    if features == None:
      orb = cv2.ORB_create()
    else:
      orb = cv2.ORB_create(features)
      
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return kp, des

book_ref_path = "a2/A2_smvs/book_covers/Reference/"
book_query_path = "a2/A2_smvs/book_covers/Query/"

book_ref = findIndex(1, 102, book_ref_path)
book_query = findIndex(1, 102, book_query_path)
book_labels = list(range(1, 102))

landmark_ref_path = "a2/A2_smvs/landmarks/Reference/"
landmark_query_path = "a2/A2_smvs/landmarks/Query/"

landmark_ref = findIndex(1, 101, landmark_ref_path)
landmark_query = findIndex(1, 101, landmark_query_path)
landmark_labels = list(range(1, 101))