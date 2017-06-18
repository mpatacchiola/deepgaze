#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import tensorflow as tf
import cv2
import os.path
import imp #to check for missing modules
import math

#Check if dlib is installed
try:
    imp.find_module('dlib')
    IS_DLIB_INSTALLED = True
    import dlib
    print('[DEEPGAZE] head_pose_estimation.py: the dlib library is installed.')
except ImportError:
    IS_DLIB_INSTALLED = False
    print('[DEEPGAZE] head_pose_estimation.py: the dlib library is not installed.')

#Enbale if you need printing utilities
DEBUG = False


class CnnHeadPoseEstimator:
    """ Head pose estimation class which uses convolutional neural network

        It finds Roll, Pitch and Yaw of the head given an head figure as input.
        It manages input (colour) picture larger than 64x64 pixels. The CNN are robust
        to variance in the input features and can handle occlusions and bad
        lighting conditions. The output values are in the ranges (degrees): 
        ROLL=[-40, +40]
        YAW=[-100, +100] 
    """

    def __init__(self, tf_session):
        """ Init the class

        @param tf_session An external tensorflow session
        """
        self._sess = tf_session

    def print_allocated_variables(self):
        """ Print all the Tensorflow allocated variables

        """
        all_vars = tf.all_variables()

        print("[DEEPGAZE] Printing all the Allocated Tensorflow Variables:")
        for k in all_vars:
            print(k.name)     

    def _allocate_yaw_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_yaw_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))
        
        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hy_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hy_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hy_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hy_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hy_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hy_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hy_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Dense layer
        #[ , num_hidden] wd2
        #self.hy_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hy_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hy_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hy_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # dropout (keep probability)
        #self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])
            if(DEBUG == True): print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hy_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv1_biases))
            if(DEBUG == True): print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hy_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv2_biases))
            if(DEBUG == True): print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hy_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hy_conv3_biases))
            if(DEBUG == True): print("SHAPE conv3: " + str(conv3.get_shape()))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool3: " + str(pool3.get_shape()))
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hy_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            if(DEBUG == True): print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.tanh(tf.matmul(dense1, self.hy_dense1_weights) + self.hy_dense1_biases)

            #Fully connected layer 5
            #dense2 = tf.tanh(tf.matmul(dense1, self.hy_dense2_weights) + self.hy_dense2_biases) 
            #if(DEBUG == True): print("SHAPE dense2: " + str(dense2.get_shape()))

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hy_out_weights) + self.hy_out_biases)
            if(DEBUG == True): print("SHAPE out: " + str(out.get_shape()))

            return out
 
        # Get the result from the model
        self.cnn_yaw_output = model(self.tf_yaw_input_vector)

 
    def load_yaw_variables(self, YawFilePath):
        """ Load varibles from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param YawFilePath Path to a valid checkpoint
        """

        #Allocate the variables in memory
        self._allocate_yaw_variables()

        #It is possible to use the checkpoint file
        #y_ckpt = tf.train.get_checkpoint_state(YawFilePath)
        #.restore(self._sess, y_ckpt.model_checkpoint_path) 

        #For future use, allocating a fraction of the GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Allocate only half of the GPU memory

        if(os.path.isfile(YawFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_yaw_variables): the yaw file path is incorrect.')

        tf.train.Saver(({"conv1_yaw_w": self.hy_conv1_weights, "conv1_yaw_b": self.hy_conv1_biases,
                         "conv2_yaw_w": self.hy_conv2_weights, "conv2_yaw_b": self.hy_conv2_biases,
                         "conv3_yaw_w": self.hy_conv3_weights, "conv3_yaw_b": self.hy_conv3_biases,
                         "dense1_yaw_w": self.hy_dense1_weights, "dense1_yaw_b": self.hy_dense1_biases,
                         "out_yaw_w": self.hy_out_weights, "out_yaw_b": self.hy_out_biases
                        })).restore(self._sess, YawFilePath) 


    def return_yaw(self, image, radians=False):
         """ Return the yaw angle associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """
         #Uncomment if you want to see the image
         #cv2.imshow('image',image)
         #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         h, w, d = image.shape
         #check if the image has the right shape
         if(h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)
             yaw_vector = np.multiply(yaw_raw, 100.0)
             #yaw = yaw_raw #* 100 #cnn out is in range [-1, +1] --> [-100, + 100]
             if(radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector
         #If the image is > 64 pixel then resize it
         if(h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_yaw_input_vector : image_normalised}
             yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)       
             yaw_vector = np.multiply(yaw_raw, 100.0) #cnn-out is in range [-1, +1] --> [-100, + 100]
             if(radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
             else: return yaw_vector
         #wrong shape          
         if(h != w or w<64 or h<64):
             if h != w :
		raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
             else:		
             	raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
         #wrong number of channels
         if(d!=3):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input does not have 3 channels, this function accepts only colour images.')

    def _allocate_pitch_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_pitch_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))
        
        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hp_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hp_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hp_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hp_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hp_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hp_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hp_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Dense layer
        #[ , num_hidden] wd2
        #self.hp_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hp_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hp_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hp_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # dropout (keep probability)
        #self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
 
        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])
            if(DEBUG == True): print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hp_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv1_biases))
            if(DEBUG == True): print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hp_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv2_biases))
            if(DEBUG == True): print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hp_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hp_conv3_biases))
            if(DEBUG == True): print("SHAPE conv3: " + str(conv3.get_shape()))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool3: " + str(pool3.get_shape()))
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hp_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            if(DEBUG == True): print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.tanh(tf.matmul(dense1, self.hp_dense1_weights) + self.hp_dense1_biases)

            #Fully connected layer 5
            #dense2 = tf.tanh(tf.matmul(dense1, self.hp_dense2_weights) + self.hp_dense2_biases) 
            #if(DEBUG == True): print("SHAPE dense2: " + str(dense2.get_shape()))

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hp_out_weights) + self.hp_out_biases)
            if(DEBUG == True): print("SHAPE out: " + str(out.get_shape()))

            return out
 
        # Get the result from the model
        self.cnn_pitch_output = model(self.tf_pitch_input_vector)

 
    def load_pitch_variables(self, pitchFilePath):
        """ Load varibles from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param pitchFilePath Path to a valid checkpoint
        """

        #Allocate the variables in memory
        self._allocate_pitch_variables()

        #It is possible to use the checkpoint file
        #y_ckpt = tf.train.get_checkpoint_state(pitchFilePath)
        #.restore(self._sess, y_ckpt.model_checkpoint_path) 

        #For future use, allocating a fraction of the GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Allocate only half of the GPU memory

        if(os.path.isfile(pitchFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_pitch_variables): the pitch file path is incorrect.')

        tf.train.Saver(({"conv1_pitch_w": self.hp_conv1_weights, "conv1_pitch_b": self.hp_conv1_biases,
                         "conv2_pitch_w": self.hp_conv2_weights, "conv2_pitch_b": self.hp_conv2_biases,
                         "conv3_pitch_w": self.hp_conv3_weights, "conv3_pitch_b": self.hp_conv3_biases,
                         "dense1_pitch_w": self.hp_dense1_weights, "dense1_pitch_b": self.hp_dense1_biases,
                         "out_pitch_w": self.hp_out_weights, "out_pitch_b": self.hp_out_biases
                        })).restore(self._sess, pitchFilePath) 


    def return_pitch(self, image, radians=False):
         """ Return the pitch angle associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """
         #Uncomment if you want to see the image
         #cv2.imshow('image',image)
         #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         h, w, d = image.shape
         #check if the image has the right shape
         if(h == w and h==64 and d==3):
             image_normalised = np.add(image, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)
             pitch_vector = np.multiply(pitch_raw, 45.0)
             #pitch = pitch_raw #* 40 #cnn out is in range [-1, +1] --> [-45, + 45]
             if(radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #If the image is > 64 pixel then resize it
         if(h == w and h>64 and d==3):
             image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
             image_normalised = np.add(image_resized, -127) #normalisation of the input
             feed_dict = {self.tf_pitch_input_vector : image_normalised}
             pitch_raw = self._sess.run([self.cnn_pitch_output], feed_dict=feed_dict)       
             pitch_vector = np.multiply(pitch_raw, 45.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
             if(radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
             else: return pitch_vector
         #wrong shape          
         if(h != w or w<64 or h<64):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_pitch): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
         #wrong number of channels
         if(d!=3):
             raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_pitch): the image given as input does not have 3 channels, this function accepts only colour images.')

    def _allocate_roll_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._num_labels = 1
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_roll_input_vector = tf.placeholder(tf.float32, shape=(64, 64, 3))

        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.hr_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
        self.hr_conv1_biases = tf.Variable(tf.zeros([64]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hr_conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.hr_conv2_biases = tf.Variable(tf.random_normal(shape=[128]))
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.hr_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)) #was[3, 3, 128, 256]
        self.hr_conv3_biases = tf.Variable(tf.random_normal(shape=[256]))

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.hr_dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1)) #was [5*5*256, 1024]
        self.hr_dense1_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Dense layer
        #[ , num_hidden] wd2
        #self.hr_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hr_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.hr_out_weights = tf.Variable(tf.truncated_normal([256, self._num_labels], stddev=0.1))
        self.hr_out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]))

        # Model.
        def model(data):

            X = tf.reshape(data, shape=[-1, 64, 64, 3])
            if(DEBUG == True): print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(X, self.hr_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hr_conv1_biases))
            if(DEBUG == True): print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)
 
            # Convolution Layer 2
            conv2 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm1, self.hr_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hr_conv2_biases))
            if(DEBUG == True): print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm2 = tf.nn.dropout(norm2, _dropout)

            # Convolution Layer 3
            conv3 = tf.tanh(tf.nn.bias_add(tf.nn.conv2d(norm2, self.hr_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),self.hr_conv3_biases))
            if(DEBUG == True): print("SHAPE conv3: " + str(conv3.get_shape()))
            # Max Pooling (down-sampling)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if(DEBUG == True): print("SHAPE pool3: " + str(pool3.get_shape()))
            # Apply Normalization
            norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # Fully connected layer 4
            dense1 = tf.reshape(norm3, [-1, self.hr_dense1_weights.get_shape().as_list()[0]]) # Reshape conv3
            if(DEBUG == True): print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.tanh(tf.matmul(dense1, self.hr_dense1_weights) + self.hr_dense1_biases)

            #Fully connected layer 5
            #dense2 = tf.tanh(tf.matmul(dense1, self.hr_dense2_weights) + self.hr_dense2_biases) 
            #if(DEBUG == True): print("SHAPE dense2: " + str(dense2.get_shape()))

            #Output layer 6
            out = tf.tanh(tf.matmul(dense1, self.hr_out_weights) + self.hr_out_biases)
            if(DEBUG == True): print("SHAPE out: " + str(out.get_shape()))

            return out
        # Get the result from the model
        self.cnn_roll_output = model(self.tf_roll_input_vector)

    def load_roll_variables(self, rollFilePath):
        """ Load variables from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param rollFilePath Path to a valid checkpoint
        """

        #Allocate the variables in memory
        self._allocate_roll_variables()

        #It is possible to use the checkpoint file
        #y_ckpt = tf.train.get_checkpoint_state(rollFilePath)
        #.restore(self._sess, y_ckpt.model_checkpoint_path) 

        #For future use, allocating a fraction of the GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) #Allocate only half of the GPU memory

        if(os.path.isfile(rollFilePath)==False): raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(load_roll_variables): the roll file path is incorrect.')

        tf.train.Saver(({"conv1_roll_w": self.hr_conv1_weights, "conv1_roll_b": self.hr_conv1_biases,
                         "conv2_roll_w": self.hr_conv2_weights, "conv2_roll_b": self.hr_conv2_biases,
                         "conv3_roll_w": self.hr_conv3_weights, "conv3_roll_b": self.hr_conv3_biases,
                         "dense1_roll_w": self.hr_dense1_weights, "dense1_roll_b": self.hr_dense1_biases,
                         "out_roll_w": self.hr_out_weights, "out_roll_b": self.hr_out_biases
                        })).restore(self._sess, rollFilePath) 


    def return_roll(self, image, radians=False):
        """ Return the roll angle associated with the input image.

        @param image It is a colour image. It must be >= 64 pixel.
        @param radians When True it returns the angle in radians, otherwise in degrees.
        """
        #Uncomment if you want to see the image
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        h, w, d = image.shape
        #check if the image has the right shape
        if(h == w and h==64 and d==3):
            image_normalised = np.add(image, -127) #normalisation of the input
            feed_dict = {self.tf_roll_input_vector : image_normalised}
            roll_raw = self._sess.run([self.cnn_roll_output], feed_dict=feed_dict)
            roll_vector = np.multiply(roll_raw, 25.0)
            #roll = roll_raw #* 40 #cnn out is in range [-1, +1] --> [-25, + 25]
            if(radians==True): return np.multiply(roll_vector, np.pi/180.0) #to radians
            else: return roll_vector
        #If the image is > 64 pixel then resize it
        if(h == w and h>64 and d==3):
            image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
            image_normalised = np.add(image_resized, -127) #normalisation of the input
            feed_dict = {self.tf_roll_input_vector : image_normalised}
            roll_raw = self._sess.run([self.cnn_roll_output], feed_dict=feed_dict)       
            roll_vector = np.multiply(roll_raw, 25.0) #cnn-out is in range [-1, +1] --> [-25, + 25]
            if(radians==True): return np.multiply(roll_vector, np.pi/180.0) #to radians
            else: return roll_vector
        #wrong shape          
        if(h != w or w<64 or h<64):
            raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_roll): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
        #wrong number of channels
        if(d!=3):
            raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_roll): the image given as input does not have 3 channels, this function accepts only colour images.')    

class PnpHeadPoseEstimator:
    """ Head pose estimation class which uses the OpenCV PnP algorithm.

        It finds Roll, Pitch and Yaw of the head given a figure as input.
        It uses the PnP algorithm and it requires the dlib library
    """

    def __init__(self, cam_w, cam_h, dlib_shape_predictor_file_path):
        """ Init the class

        @param cam_w the camera width. If you are using a 640x480 resolution it is 640
        @param cam_h the camera height. If you are using a 640x480 resolution it is 480
        @dlib_shape_predictor_file_path path to the dlib file for shape prediction (look in: deepgaze/etc/dlib/shape_predictor_68_face_landmarks.dat)
        """
        if(IS_DLIB_INSTALLED == False): raise ValueError('[DEEPGAZE] PnpHeadPoseEstimator: the dlib libray is not installed. Please install dlib if you want to use the PnpHeadPoseEstimator class.')
        if(os.path.isfile(dlib_shape_predictor_file_path)==False): raise ValueError('[DEEPGAZE] PnpHeadPoseEstimator: the files specified do not exist.') 

        #Defining the camera matrix.
        #To have better result it is necessary to find the focal
        # lenght of the camera. fx/fy are the focal lengths (in pixels) 
        # and cx/cy are the optical centres. These values can be obtained 
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60/2 * np.pi / 180)
        f_y = f_x

        #Estimated camera matrix values.
        self.camera_matrix = np.float32([[f_x, 0.0, c_x],
                                         [0.0, f_y, c_y], 
                                         [0.0, 0.0, 1.0] ])
        #These are the camera matrix values estimated on my webcam with
        # the calibration code (see: src/calibration):
        #camera_matrix = np.float32([[602.10618226,          0.0, 320.27333589],
                                   #[         0.0, 603.55869786,  229.7537026], 
                                   #[         0.0,          0.0,          1.0] ])

        #Distortion coefficients
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        #Distortion coefficients estimated by calibration in my webcam
        #camera_distortion = np.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])

        if(DEBUG==True): print("[DEEPGAZE] PnpHeadPoseEstimator: estimated camera matrix: \n" + str(self.camera_matrix) + "\n")

        #Declaring the dlib shape predictor object
        self._shape_predictor = dlib.shape_predictor(dlib_shape_predictor_file_path)


    def _return_landmarks(self, inputImg, roiX, roiY, roiW, roiH, points_to_return=range(0,68)):
        """ Return the the roll pitch and yaw angles associated with the input image.

        @param image It is a colour image. It must be >= 64 pixel.
        @param radians When True it returns the angle in radians, otherwise in degrees.
        """
        #Creating a dlib rectangle and finding the landmarks
        dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
        dlib_landmarks = self._shape_predictor(inputImg, dlib_rectangle)

        #It selects only the landmarks that
        #have been indicated in the input parameter "points_to_return".
        #It can be used in solvePnP() to estimate the 3D pose.
        landmarks = np.zeros((len(points_to_return),2), dtype=np.float32)
        counter = 0
        for point in points_to_return:
            landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
            counter += 1

        return landmarks

    def return_roll_pitch_yaw(self, image, radians=False):
         """ Return the the roll pitch and yaw angles associated with the input image.

         @param image It is a colour image. It must be >= 64 pixel.
         @param radians When True it returns the angle in radians, otherwise in degrees.
         """

         #The dlib shape predictor returns 68 points, we are interested only in a few of those
         TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

         #Antropometric constant values of the human head. 
         #Check the wikipedia EN page and:
         #"Head-and-Face Anthropometric Survey of U.S. Respirator Users"
         #
         #X-Y-Z with X pointing forward and Y on the left and Z up.
         #The X-Y-Z coordinates used are like the standard
         # coordinates of ROS (robotic operative system)
         #OpenCV uses the reference usually used in computer vision: 
         #X points to the right, Y down, Z to the front
         #
         #The Male mean interpupillary distance is 64.7 mm (https://en.wikipedia.org/wiki/Interpupillary_distance)
         #
         P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
         P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
         P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
         P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
         P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
         P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
         P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
         P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27 This is the world origin
         P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
         P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
         P3D_RIGHT_EYE = np.float32([-20.0, -32.35,-5.0]) #36 
         P3D_RIGHT_TEAR = np.float32([-10.0, -20.25,-5.0]) #39
         P3D_LEFT_TEAR = np.float32([-10.0, 20.25,-5.0]) #42
         P3D_LEFT_EYE = np.float32([-20.0, 32.35,-5.0]) #45
         #P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
         #P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
         P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62

         #This matrix contains the 3D points of the
         # 11 landmarks we want to find. It has been
         # obtained from antrophometric measurement
         # of the human head.
         landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

         #Return the 2D position of our landmarks
         img_h, img_w, img_d = image.shape
         landmarks_2D = self._return_landmarks(inputImg=image, roiX=0, roiY=img_w, roiW=img_w, roiH=img_h, points_to_return=TRACKED_POINTS)

         #Print som red dots on the image       
         #for point in landmarks_2D:
             #cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


         #Applying the PnP solver to find the 3D pose
         #of the head from the 2D position of the
         #landmarks.
         #retval - bool
         #rvec - Output rotation vector that, together with tvec, brings 
         #points from the world coordinate system to the camera coordinate system.
         #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
         retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                           landmarks_2D, 
                                           self.camera_matrix, 
                                           self.camera_distortion)

         #Get as input the rotational vector
         #Return a rotational matrix
         rmat, _ = cv2.Rodrigues(rvec) 

         #euler_angles contain (pitch, yaw, roll)
         #euler_angles = cv.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)

         head_pose = [ rmat[0,0], rmat[0,1], rmat[0,2], tvec[0],
                       rmat[1,0], rmat[1,1], rmat[1,2], tvec[1],
                       rmat[2,0], rmat[2,1], rmat[2,2], tvec[2],
                             0.0,      0.0,        0.0,    1.0 ]
         #print(head_pose) #TODO remove this line
         return self.rotationMatrixToEulerAngles(rmat)



    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :
 
        #assert(isRotationMatrix(R))
     
        #To prevent the Gimbal Lock it is possible to use
        #a threshold of 1e-6 for discrimination
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])    
        singular = sy < 1e-6
 
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
 
        return np.array([x, y, z])






