Updates
----------

**Update 22/01/2020** You may be interested in following [my new Youtube channel](https://www.youtube.com/channel/UC6AxKVw2y_b3ab-esLdK0_g) for weekly videos about *Computer Vision*, *Machine Learning*, *Deep Learning*, and *Robotics*.

**Update 16/07/2019** Stable version of *Deepgaze 2.0* is available on branch `2.0`.

**Update 20/03/2019** Started the porting on Python/OpenCV 3.0, check the branch `2.0` for a preliminary version.

**Update 10/06/2017** The PDF of the article *"Head pose estimation in the wild using Convolutional Neural Networks and adaptive gradient methods"* is available for **free download** in the next 50 days using [this special link](https://authors.elsevier.com/a/1VBdC77nKOnOt)

**Update 04/06/2017** Article *"Head pose estimation in the wild using Convolutional Neural Networks and adaptive gradient methods"* have been accepted for publication in Pattern Recogntion (Elsevier). The Deepgaze CNN head pose estimator module is based on this work.

**Update 31/05/2017** Implementation of the new package [saliency_map.py](./deepgaze/saliency_map.py). The package contains an implementation of the [FASA](http://ivrl.epfl.ch/research/saliency/fast_saliency) algorithm for saliency detection [[example]](./examples/ex_fasa_saliency_map/ex_fasa_saliency_map_images.py) [[wiki]](http://www.scholarpedia.org/article/Saliency_map)

**Update 22/03/2017** Fixed a bug in mask_analysis.py and almost completed a more robust version of the CNN head pose estimator.

What is Deepgaze?
----------
Deepgaze is a library for human-computer interaction, people detection and tracking which uses **Convolutional Neural Networks** (CNNs) for face detection, head pose estimation and classification. The focus of attention of a person can be approximately estimated finding the **head orientation**. This is particularly useful when the eyes are covered, or when the user is too far from the camera to grab the eye region with a good resolution. When the eye region is visible it is possible to estimate the **gaze direction**, which is much more informative and can give a good indication of the FOA. Deepgaze contains useful packages for:

- Head pose estimation (Perspective-n-Point, Convolutional Neural Networks)
- Face detection (Haar Cascade)
- Skin and color detection (Range detection, Backprojection)
- Histogram-based classification (Histogram Intersection)
- Motion detection (Frame differencing, MOG, MOG2)
- Motion tracking (Particle filter)
- Saliency map (FASA)

Deepgaze is based on OpenCV and Tensorflow, some of the best libraries in computer vision and machine learning. Deepgaze is an **open source** project and any contribution is appreciated, feel free to fork the repository and propose integrations. 

This library is the result of a recent work, **if you use the library in academic work please cite the following paper**:

Patacchiola, M., & Cangelosi, A. (2017). *Head pose estimation in the wild using Convolutional Neural Networks and adaptive gradient methods*. Pattern Recognition, http://dx.doi.org/10.1016/j.patcog.2017.06.009.

Why should I use Deepgaze?
--------------------------
Because Deepgaze **makes your life easier!**
The implementation of many algorithms such as face detectors, pose estimators and object classificators can be painful. Deepgaze has been designed to implement those algorithms in **a few lines of code**. Deepgaze is helpful for both beginners and advanced users who want to save time. All the code contained in Deepgaze is optimised and it is based on state-of-the-art algorithms.

What is a Convolutional Neural Network?
------------------------------
A convolutional neural network (CNN, or ConvNet) is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, whose individual neurons are arranged in such a way that they respond to overlapping regions tiling the visual field. Convolutional networks were inspired by biological processes and are variations of multilayer perceptrons designed to use minimal amounts of preprocessing. They have wide applications in image and video recognition, recommender systems and natural language processing [[wiki]](https://en.wikipedia.org/wiki/Convolutional_neural_network)

<p align="center">
<img src="doc/images/figure_cnn.png" width="750">
</p>

Main contributors
-------------------
This is an updated list of the **main contributors** of the project. **We are looking for contributors!** If you want to contribute adding a new module or improving an existing one, [send an email to our team!](https://www.inf.ed.ac.uk/people/staff/Massimiliano_Patacchiola.html)

- [Massimiliano Patacchiola](http://mpatacchiola.github.io/): project leader and main contributor
- [Joel Gooch](https://www.linkedin.com/in/joel-gooch-001458132/?ppe=1): head pose estimation
- [Ishit Mehta](https://github.com/ishit): CNN-cascade face detection
- [Luca Surace](https://github.com/lukeoverride): Haar-cascade multi-face detection
- [Hrishikesh Kamath](https://github.com/kamathhrishi): version 2.0 porting, notebooks, test scripts

Prerequisites
------------

The current version of Deepgaze is based on **Python 2.7**, a porting for Python 3.0 has been scheduled for the next year.

(Recommended) Create a Python virtual environment:
You can follow the [Python official documentation](https://docs.python.org/3/tutorial/venv.html) for virtual environments .

To use the libray you have to install:

- Numpy [[link]](http://www.numpy.org/)

```shell
sudo pip install numpy
```

- OpenCV 2.x (not compatible with OpenCV >= 3.x) [[link]](http://opencv.org/)

```shell
sudo apt-get install libopencv-dev python-opencv
```

- Tensorflow [[link]](https://www.tensorflow.org/)

```shell
sudo pip install tensorflow
```

Some examples may require additional libraries:

- dlib [[link]](http://dlib.net/)

Installation
--------

ATTENTION: this version is obsolete, please check the [branch 2.0 on this repository](https://github.com/mpatacchiola/deepgaze/tree/2.0)

Download the repository from [[here]](https://github.com/mpatacchiola/deepgaze/archive/master.zip) or clone it using git:

```shell
git clone https://github.com/mpatacchiola/deepgaze.git
```

To install the package you have to enter in the Deepgaze folder and run the setup.py script (it may require root privileges):

```shell
cd deepgaze
sudo python setup.py install
```

If you want to track all the installed files you can record the installation process in a text file using the `--record` flag:

```shell
sudo python setup.py install --record record.txt
```

Done! Now give a look to the examples below.

Examples
--------

- Head Pose Estimation using the Perspective-n-Point algorithm in OpenCV [[code]](./examples/ex_pnp_head_pose_estimation_webcam.py) [[video]](https://www.youtube.com/watch?v=OSnI18XmAg4)

- Head Pose Estimation in-the-wild using Perspective-n-Point and dlib face detector [[code]](./examples/ex_dlib_pnp_head_pose_estimation_video.py) [[video]](https://www.youtube.com/watch?v=xurEs0G9ARs)

- Head Pose Estimation in images using Convolutional Neural Networks [[code]](./examples/ex_cnn_head_pose_estimation_images/ex_cnn_head_pose_estimation_images.py)

<p align="center">
<img src="doc/images/ex_cnn_head_pose_estimation_images.png" width="750">
</p>

- Color detection using the Histogram Backprojection algorithm [[blog]](https://mpatacchiola.github.io/blog/2016/12/01/playing-the-google-chrome-dinosaur-game-with-your-hand.html) [[code]](./examples/ex_color_detection_image/ex_color_detection_image.py)

<p align="center">
<img src="doc/images/ex_color_detection_image.png" width="750">
</p>

- Skin detection using the HSV range color detector [[code]](./examples/ex_skin_detection_images/ex_skin_detection_images.py)

<p align="center">
<img src="doc/images/ex_skin_detection_images.png" width="750">
</p>

- Face detection using the HSV range color detector [[code]](./examples/ex_face_center_color_detection/ex_face_center_color_detection.py)

<p align="center">
<img src="doc/images/ex_face_center_color_detection.png" width="750">
</p>

- Corner detection comparison of four algorithms on a video streaming [[code]](./examples/ex_corner_detection_video/ex_corner_detection.py) [[video]](https://www.youtube.com/watch?v=2fhD98K_6Ag)

<p align="center">
<img src="doc/images/ex_corner_detection.png" width="750">
</p>

- Motion detection and tracking using frame differencing on a video streaming [[code]](./examples/ex_diff_motion_detection_video/ex_diff_motion_detection.py)

<p align="center">
<img src="doc/images/ex_diff_motion_detection_video.png" width="750">
</p>

- Motion detection and tracking comparison of three algorithms on a video streaming [[code]](./examples/ex_motion_detectors_comparison_video/ex_motion_detectors_comparison_video.py) [[video]](https://www.youtube.com/watch?v=XmI2kE2hUgE)

<p align="center">
<img src="doc/images/ex_motion_detectors_comparison_video.png" width="750">
</p>

- Motion tracking with unstable measurements using Particle Filter [[code]](./examples/ex_particle_filter_object_tracking_video/ex_particle_filter_object_tracking_video.py) [[video]](https://www.youtube.com/watch?v=KTxVBN5-KpE)
<p align="center">
<img src="doc/images/ex_particle_filtering_object_tracking_video.png" width="750">
</p>

- Motion tracking with multiple backprojection for playing chrome's dinosaur game [[blog]](https://mpatacchiola.github.io/blog/2016/12/01/playing-the-google-chrome-dinosaur-game-with-your-hand.html) [[code]](./examples/ex_multi_backprojection_hand_tracking_gaming/ex_multi_backprojection_hand_tracking_gaming.py) [[video]](https://www.youtube.com/watch?v=eoUOkV5vVpU&feature=youtu.be)
<p align="center">
<img src="doc/images/ex_multi_backprojection_hand_tracking_gaming.gif" width="550">
</p>

- Classify object using their colour fingerprint (histogram intersection) [[blog]](https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html) [[code]](./examples/ex_color_classification_images/ex_color_classification_image.py)
<p align="center">
<img src="doc/images/ex_color_classification_images.png" width="750">
</p>

- Implementation of the FASA (Fast, Accurate, and Size-Aware Salient Object Detection) algorithm [[code]](./examples/ex_fasa_saliency_map/ex_fasa_saliency_map_images.py) [[wiki]](http://www.scholarpedia.org/article/Saliency_map) [[link]](http://ivrl.epfl.ch/research/saliency/fast_saliency)
<p align="center">
<img src="doc/images/ex_fasa_saliency_map.png" width="750">
</p>

Acknowledgements
---------------

- The example "head pose estimation using Perspective-n-Point" is partially based on the C++ version you can find [here](https://github.com/severin-lemaignan/gazr), and on the workshop "Developing an attention system for a social robot" which was part of the 2nd International Summer School on Social Human-Robot Interaction.

- To implement the Bayes and Particle Filters I followed the great repository of [rlabbe](https://github.com/rlabbe) which you can find [here](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)









