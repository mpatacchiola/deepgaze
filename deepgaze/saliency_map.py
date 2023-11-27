#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io
# https://mpatacchiola.github.io/blog/
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
import sys
from timeit import default_timer as timer

DEBUG = False

class FasaSaliencyMapping:
    """Implementation of the FASA (Fast, Accurate, and Size-Aware Salient Object Detection) algorithm.

    Abstract:
    Fast and accurate salient-object detectors are important for various image processing and computer vision 
    applications, such as adaptive compression and object segmentation. It is also desirable to have a detector that is 
    aware of the position and the size of the salient objects. In this paper, we propose a salient-object detection 
    method that is fast, accurate, and size-aware. For efficient computation, we quantize the image colors and estimate 
    the spatial positions and sizes of the quantized colors. We then feed these values into a statistical model to 
    obtain a probability of saliency. In order to estimate the final saliency, this probability is combined with a 
    global color contrast measure. We test our method on two public datasets and show that our method significantly 
    outperforms the fast state-of-the-art methods. In addition, it has comparable performance and is an order of 
    magnitude faster than the accurate state-of-the-art methods. We exhibit the potential of our algorithm by 
    processing a high-definition video in real time. 
    """

    def __init__(self, image_h, image_w):
        """Init the classifier.

        """
        # Assigning some global variables and creating here the image to fill later (for speed purposes)
        self.image_rows = image_h
        self.image_cols = image_w
        self.salient_image = np.zeros((image_h, image_w), dtype=np.uint8)
        # mu: mean vector
        self.mean_vector = np.array([0.5555, 0.6449, 0.0002, 0.0063])
        # covariance matrix
        # self.covariance_matrix = np.array([[0.0231, -0.0010, 0.0001, -0.0002],
        #                                    [-0.0010, 0.0246, -0.0000, 0.0000],
        #                                    [0.0001, -0.0000, 0.0115, 0.0003],
        #                                    [-0.0002, 0.0000, 0.0003, 0.0080]])
        # determinant of covariance matrix
        # self.determinant_covariance = np.linalg.det(self.covariance_matrix)
        # self.determinant_covariance = 5.21232874e-08
        # Inverse of the covariance matrix
        self.covariance_matrix_inverse = np.array([[43.3777, 1.7633, -0.4059, 1.0997],
                                                   [1.7633, 40.7221, -0.0165, 0.0447],
                                                   [-0.4059, -0.0165, 87.0455, -3.2744],
                                                   [1.0997, 0.0447, -3.2744, 125.1503]])

    def _calculate_histogram(self, image, tot_bins=8):
        # 1- Conversion from BGR to LAB color space
        # Here a color space conversion is done. Moreover the min/max value for each channel is found.
        # This is helpful because the 3D histogram will be defined in this sub-space.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        minL, maxL, _, _ = cv2.minMaxLoc(image[:, :, 0])
        minA, maxA, _, _ = cv2.minMaxLoc(image[:, :, 1])
        minB, maxB, _, _ = cv2.minMaxLoc(image[:, :, 2])

        # Quantization ranges
        self.L_range = np.linspace(minL, maxL, num=tot_bins, endpoint=False)
        self.A_range = np.linspace(minA, maxA, num=tot_bins, endpoint=False)
        self.B_range = np.linspace(minB, maxB, num=tot_bins, endpoint=False)
        # Here the image quantized using the discrete bins is created.
        self.image_quantized = np.dstack((np.digitize(image[:, :, 0], self.L_range, right=False),
                                          np.digitize(image[:, :, 1], self.A_range, right=False),
                                          np.digitize(image[:, :, 2], self.B_range, right=False)))
        self.image_quantized -= 1  # now in range [0,7]

        # it maps the 3D index of hist in a flat 1D array index
        self.map_3d_1d = np.zeros((tot_bins, tot_bins, tot_bins), dtype=np.int32)

        # Histograms in a 3D manifold of shape (tot_bin, tot_bin, tot_bin).
        # The cv2.calcHist for a 3-channels image generates a cube of size (tot_bins, tot_bins, tot_bins) which is a
        # discretization of the 3-D space defined by hist_range.
        # E.G. if range is 0-255 and it is divided in 5 bins we get -> [0-50][50-100][100-150][150-200][200-250]
        # So if you access the histogram with the indeces: histogram[3,0,2] it is possible to see how many pixels
        # fall in the range channel_1=[150-200], channel_2=[0-50], channel_3=[100-150]
        # data = np.vstack((image[:, :, 0].flat, image[:, :, 1].flat, image[:, :, 2].flat)).astype(np.uint8).T
        # OpenCV implementation is slightly faster than Numpy
        self.histogram = cv2.calcHist([image], channels=[0, 1, 2], mask=None,
                                      histSize=[tot_bins, tot_bins, tot_bins],
                                      ranges=[minL, maxL, minA, maxA, minB, maxB])
        # data = np.vstack((image[:, :, 0].flat, image[:, :, 1].flat, image[:, :, 2].flat)).T
        # self.histogram, edges = np.histogramdd(data, bins=tot_bins, range=((minL, maxL), (minA, maxA), (minB, maxB)))
        # self.histogram, edges = np.histogramdd(data, bins=tot_bins)

        # Get flatten index ID of the image pixels quantized
        image_indeces = np.vstack((self.image_quantized[:, :, 0].flat,
                                   self.image_quantized[:, :, 1].flat,
                                   self.image_quantized[:, :, 2].flat)).astype(np.int32)
        image_linear = np.ravel_multi_index(image_indeces, (tot_bins, tot_bins, tot_bins))  # in range [0,7]
        # image_linear = np.reshape(image_linear, (self.image_rows, self.image_cols))
        # Getting the linear ID index of unique colours
        self.index_matrix = np.transpose(np.nonzero(self.histogram))
        hist_index = np.where(self.histogram > 0)  # Included in [0,7]
        unique_color_linear = np.ravel_multi_index(hist_index, (tot_bins, tot_bins, tot_bins))  # linear ID index
        self.number_of_colors = np.amax(self.index_matrix.shape)
        self.centx_matrix = np.zeros(self.number_of_colors)
        self.centy_matrix = np.zeros(self.number_of_colors)
        self.centx2_matrix = np.zeros(self.number_of_colors)
        self.centy2_matrix = np.zeros(self.number_of_colors)
        # Using the numpy method where() to find the location of each unique colour in the linear ID matrix
        counter = 0
        for i in unique_color_linear:
            # doing only one call to a flat image_linear is faster here
            where_y, where_x = np.unravel_index(np.where(image_linear == i), (self.image_rows, self.image_cols))
            #where_x = np.where(image_linear == i)[1]  # columns coord
            #where_y = np.where(image_linear == i)[0]  # rows coord
            self.centx_matrix[counter] = np.sum(where_x)
            self.centy_matrix[counter] = np.sum(where_y)
            self.centx2_matrix[counter] = np.sum(np.power(where_x, 2))
            self.centy2_matrix[counter] = np.sum(np.power(where_y, 2))
            counter += 1
        return image

    def _precompute_parameters(self, sigmac=16):
        """ Semi-Vectorized version of the precompute parameters function.
        This function runs at 0.003 seconds on a squared 400x400 pixel image.
        It returns the number of colors and estimates the color_distance matrix
        
        @param sigmac: the scalar used in the exponential (default=16) 
        @return: the number of unique colors
        """
        L_centroid, A_centroid, B_centroid = np.meshgrid(self.L_range, self.A_range, self.B_range)
        self.unique_pixels = np.zeros((self.number_of_colors, 3))
        
        if sys.version_info[0] == 2:
            color_range = xrange(0, self.number_of_colors)
        else:
            color_range = range(0, self.number_of_colors)
        
        for i in color_range:
            i_index = self.index_matrix[i, :]
            L_i = L_centroid[i_index[0], i_index[1], i_index[2]]
            A_i = A_centroid[i_index[0], i_index[1], i_index[2]]
            B_i = B_centroid[i_index[0], i_index[1], i_index[2]]
            self.unique_pixels[i] = np.array([L_i, A_i, B_i])
            self.map_3d_1d[i_index[0], i_index[1], i_index[2]] = i  # the map is assigned here for performance purposes
        color_difference_matrix = np.sum(np.power(self.unique_pixels[:, np.newaxis] - self.unique_pixels, 2), axis=2)
        self.color_distance_matrix = np.sqrt(color_difference_matrix)
        self.exponential_color_distance_matrix = np.exp(- np.divide(color_difference_matrix, (2 * sigmac * sigmac)))
        return self.number_of_colors

    def _bilateral_filtering(self):
        """ Applying the bilateral filtering to the matrices.
        
        This function runs at 0.0006 seconds on a squared 400x400 pixel image.
        Since the trick 'matrix[ matrix > x]' is used it would be possible to set a threshold
        which is an energy value, considering only the histograms which have enough colours.
        @return: mx, my, Vx, Vy
        """
        # Obtaining the values through vectorized operations (very efficient)
        self.contrast = np.dot(self.color_distance_matrix, self.histogram[self.histogram > 0])
        normalization_array = np.dot(self.exponential_color_distance_matrix, self.histogram[self.histogram > 0])
        self.mx = np.dot(self.exponential_color_distance_matrix, self.centx_matrix)
        self.my = np.dot(self.exponential_color_distance_matrix, self.centy_matrix)
        mx2 = np.dot(self.exponential_color_distance_matrix, self.centx2_matrix)
        my2 = np.dot(self.exponential_color_distance_matrix, self.centy2_matrix)
        # Normalizing the vectors
        self.mx = np.divide(self.mx, normalization_array)
        self.my = np.divide(self.my, normalization_array)
        mx2 = np.divide(mx2, normalization_array)
        my2 = np.divide(my2, normalization_array)
        self.Vx = np.absolute(np.subtract(mx2, np.power(self.mx, 2))) # TODO: understand why some negative values appear
        self.Vy = np.absolute(np.subtract(my2, np.power(self.my, 2)))
        return self.mx, self.my, self.Vx, self.Vy

    def _calculate_probability(self):
        """ Vectorized version of the probability estimation.
        
        This function runs at 0.0001 seconds on a squared 400x400 pixel image.
        @return: a vector shape_probability of shape (number_of_colors)
        """
        g = np.array([np.sqrt(12 * self.Vx) / self.image_cols,
                      np.sqrt(12 * self.Vy) / self.image_rows,
                      (self.mx - (self.image_cols / 2.0)) / float(self.image_cols),
                      (self.my - (self.image_rows / 2.0)) / float(self.image_rows)])
        X = (g.T - self.mean_vector)
        Y = X
        A = self.covariance_matrix_inverse
        result = (np.dot(X, A) * Y).sum(1)  # This line does the trick
        self.shape_probability = np.exp(- result / 2)
        return self.shape_probability

    def _compute_saliency_map(self):
        """ Fast vectorized version of the saliency map estimation.
        
        This function runs at 7.7e-05 seconds on a squared 400x400 pixel image.
        @return: the saliency vector 
        """
        # Vectorized operations for saliency vector estimation
        self.saliency = np.multiply(self.contrast, self.shape_probability)
        a1 = np.dot(self.exponential_color_distance_matrix, self.saliency)
        a2 = np.sum(self.exponential_color_distance_matrix, axis=1)
        self.saliency = np.divide(a1, a2)
        # The saliency vector is renormalised in range [0-255]
        minVal, maxVal, _, _ = cv2.minMaxLoc(self.saliency)
        self.saliency = self.saliency - minVal
        self.saliency = 255 * self.saliency / (maxVal - minVal) + 1e-3
        return self.saliency


    def returnMask(self, image, tot_bins=8, format='BGR2LAB'):
        """ Return the saliency mask of the input image.
        
        @param: image the image to process
        @param: tot_bins the number of bins used in the histogram
        @param: format conversion, it can be one of the following:
            BGR2LAB, BGR2RGB, RGB2LAB, RGB, BGR, LAB
        @return: the saliency mask
        """
        if format == 'BGR2LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif format == 'BGR2RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif format == 'RGB2LAB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif format == 'RGB' or format == 'BGR' or format == 'LAB':
            pass
        else:
            raise ValueError('[DEEPGAZE][SALIENCY-MAP][ERROR] the input format of the image is not supported.')
        if DEBUG: start = timer()
        self._calculate_histogram(image, tot_bins=tot_bins)
        if DEBUG: end = timer()
        if DEBUG: print("--- %s calculate_histogram seconds ---" % (end - start))
        if DEBUG: start = timer()
        number_of_colors = self._precompute_parameters()
        if DEBUG: end = timer()
        if DEBUG: print("--- number of colors: " + str(number_of_colors) + " ---")
        if DEBUG: print("--- %s precompute_paramters seconds ---" % (end - start))
        if DEBUG: start = timer()
        self._bilateral_filtering()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s bilateral_filtering seconds ---" % (end - start))
        if DEBUG: start = timer()
        self._calculate_probability()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s calculate_probability seconds ---" % (end - start))
        if DEBUG: start = timer()
        self._compute_saliency_map()
        if DEBUG: end = timer()
        if DEBUG: print("--- %s compute_saliency_map seconds ---" % (end - start))
        if DEBUG: start = timer()
        it = np.nditer(self.salient_image, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            # This part takes 0.1 seconds
            y = it.multi_index[0]
            x = it.multi_index[1]
            #L_id = self.L_id_matrix[y, x]
            #A_id = self.A_id_matrix[y, x]
            #B_id = self.B_id_matrix[y, x]
            index = self.image_quantized[y, x]
            # These operations take 0.1 seconds
            index = self.map_3d_1d[index[0], index[1], index[2]]
            it[0] = self.saliency[index]
            it.iternext()

        if DEBUG: end = timer()
        #ret, self.salient_image = cv2.threshold(self.salient_image, 150, 255, cv2.THRESH_BINARY)
        if DEBUG: print("--- %s returnMask 'iteration part' seconds ---" % (end - start))
        return self.salient_image

