
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

import skimage.io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    # out = None
    # YOUR CODE HERE
    image = skimage.io.imread(img_path)  # skimage.io.imread() to load the picture
    out = image / 255.0
    
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE
    size = np.shape(image)  # Used to get the dimensional information of an array image

    print("height: ", size[0])
    print("width: ", size[1])
    
    if len(np.shape(image)) != 3:   # greyscale image
        print("channel: 1")
    else:
        print("channel: ", size[2])
    
    return None

def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_row: start_row+num_rows, start_col: start_col+num_cols, :]
    return out

def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE

    newRow = np.shape(image)[0]
    newCol = np.shape(image)[1]
    #All-zero array ‘out’ to store the new image
    out = np.zeros((newRow, newCol, 3))
    
    for row in range(newRow):
        for col in range(newCol):
            # get the pixel value from image
            x_p = image[row][col]

            for i in range(len(x_p)):
                out[row][col][i] = (factor * (x_p[i] - 0.5)) + 0.5

    return out

def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """

    ### YOUR CODE HERE
    input_rows = int(np.shape(input_image)[0])
    input_cols = int(np.shape(input_image)[1])
    
    newRow = output_rows
    newCol = output_cols

    # All-zero array ‘out’ to store the new image
    out = np.zeros((newRow, newCol, 3))
    
    for row in range(newRow):
        for col in range(newCol):
            
            row_n = int(row * (input_rows / output_rows))
            col_n = int(col * (input_cols / output_cols))
            
            # insert to the new array
            out[row][col] = input_image[row_n][col_n]

    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = None

    newRow = np.shape(input_image)[0]
    newCol = np.shape(input_image)[1]
    #All-zero array ‘out’ to store the new image
    out = np.zeros((newRow, newCol, 3))

    # Iterate over each pixel in the input image
    for row in range(newRow):
        for col in range(newCol):

            R, G, B = input_image[row, col, 0], input_image[row, col, 1], input_image[row, col, 2]

            # Calculate the average of the RGB channel values
            grayscale_value = (R + G + B) / 3.0
            out[row, col] = grayscale_value

    return out

def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    ### YOUR CODE HERE
    
    # the row of the image
    newRow = np.shape(grey_img)[0]
    # the column of the image
    newCol = np.shape(grey_img)[1]
    out = np.zeros((newRow, newCol))
    
    for row in range(newRow):
        for col in range(newCol):
            # x_n = 1, if x_p > th
            if (grey_img[row][col] > th).any():
                out[row][col] = 1

    return out

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None
    ### YOUR CODE HERE

    # shape (Hi, Wi)
    newRow = np.shape(image)[0]
    newCol = np.shape(image)[1]
    
    # get the row of kernel
    kernelRow = np.shape(kernel)[0]
    # calculate the padding pixels
    padding = int(np.floor(kernelRow / 2))
    # new image after adding padding
    newPadded = np.zeros((newRow + (2 * padding), newCol + (2 * padding)))

    # copy the image to the newpadded image
    newPaddedRow = newRow + padding
    newPaddedCol = newCol + padding
    newPadded[padding: newPaddedRow, padding: newPaddedCol] = image

    # flip the kernel
    kernel = np.flip(kernel, axis=1)
    kernel = np.flip(kernel, axis=0)

    # out: numpy array of shape (Hi, Wi)
    out = np.zeros((newRow, newCol))
    
    # iterate through image
    for row in range(padding, newPaddedRow):
        for col in range(padding, newPaddedCol):
            # the value of each pixel position in the image
            out[row - padding][col - padding] = np.sum(kernel * newPadded[row-padding: row + padding + 1, col-padding: col + padding + 1])

    return out

    


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    newRow = np.shape(image)[0]
    newCol = np.shape(image)[1]
    out = np.zeros((newRow, newCol, 3))

    # dimension of image
    Dimensionality = len(np.shape(image))

    # greyscale image
    if Dimensionality == 2:
        return conv2D(image, kernel)
    
    # RGB image
    for channel in range(3):
        new = conv2D(image[:,:,channel], kernel)
        # each channel is operated separately
        for row in range(newRow):
            for col in range(newCol):
                out[row][col][channel] = new[row][col]
        
    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    return out

