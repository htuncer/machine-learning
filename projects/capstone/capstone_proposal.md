# Machine Learning Engineer Nanodegree
## Capstone Proposal
Hasan Tuncer

August 13th, 2017

## Proposal

### Domain Background
_(approx. 1-2 paragraphs)_

My goal is to create a machine learning pipeline for detecting a vehicle on a road from a video. The video is captured by a forward looking camera mounted on a vehicle.

Self-driving cars require detection of surrounding objects - one of them is other vehicle. There are two main aproaches for detecting objects in self-driving car domain:

1) Detecting the objects from images captured by camera. Elon Musk and comma.ai are to name few believing in this approach. is  is in favor of (https://www.youtube.com/watch?v=zIwLWfaAg-8&feature=youtu.be&t=14m48s)

2) Detecting the objects from point clouds captured using Light Detection and Ranging (LIDAR) technology. Most of the self-driving car startups bets on LiDar technology such as Waymo. 

In this project, I will follow the first approach due to available extensive data and benchmark. Detecting an object from an image is one of the main reaseach areas of computer vision. Deformable part models (DPMs) and convolutional neural networks (CNNs) are two widely used distinct approaches.   Deformable Parts Models (DPM) are graphical models (Markov random fields) and use a sliding window approach where the classifier such as SVM is run at evenly spaced locations on the image. CNNs are nonlinear classfiers. CNNs are more popular due to good performance on object detection [1](https://arxiv.org/pdf/1409.5403.pdf). To illustrate,  Region-based convolutional neural networks (R-CNN) [](https://arxiv.org/abs/1311.2524) trains CNNs end-to-end to classify the proposal regions into object categories or background. R-CNN deploys region proposal algorithms (such as [EdgeBoxes](https://www.microsoft.com/en-us/research/publication/edge-boxes-locating-object-proposals-from-edges/) and [Selective Search](https://www.koen.me/research/pub/vandesande-iccv2011.pdf)) to select the region  mainly as a pre-processing step before running the CNN as classifier. [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) 
uses CNN both for the region proposal and also for the prediction.

[Google Inception](https://arxiv.org/abs/1409.4842) uses Faster R-CNN with [ResNet](https://arxiv.org/abs/1512.03385). I'd like to leverage [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) which outperforms ResNet 



[YOLO](https://arxiv.org/abs/1506.02640) frames object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.  Region proposal methods [4] (https://link.springer.com/article/10.1007/s11263-013-0620-5) generate potential bounding boxes in an image and then run a classifier on these proposed boxes.

Use this: https://flyyufelix.github.io/2017/04/16/kaggle-nature-conservancy.html

However, state of art models use deep learning such as [Google Inception](https://arxiv.org/abs/1409.4842), 

YOLO .. https://github.com/subodh-malgonde/vehicle-detection

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_
Self-driving cars need to identify objects around them such as other vehicles on the road. In this problem, the objects are captured by a forward looking camera mounted on a vehicle. Identification of a vehicle will be important factor in deciding the next action that self-driving car will take such as changing lane.

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).



### Datasets and Inputs
_(approx. 2-3 paragraphs)_
To train my model, I will use the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) which are retrieved by Udacity from  [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), [the KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). If I find the training data set is not enough then I will use the labeled data [here](https://github.com/udacity/self-driving-car/tree/master/annotations).

I will run my pipeline on [the video stream](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/project_video.mp4) provided by Udacity.

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

Faster R-CNN will be used for object region detection. DenseNet will be used for classifying if the detected object is a car or not.See Project  Design section for details.

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_


* Load the data set: The training data mentioned above will be loaded. I will be careful to use the same number of vehicle and non-vehicle image. I will check the number of different labels; how different the labels among each other; distribution of images per label; and  if there is any mis-labeled image.
type of What are the different type of vehicles exist in the training set. The characteristics of the data will be explored. 

* Design the model: 
  * Object detection: I will use [pre-trained faster r-cnn inception resnetv2] (https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for region detection. It will take days to train a model from scratch. The reason of selecting faster r-cnn is its high accuracy result [](https://arxiv.org/abs/1611.10012). 
  
  * Object classification: I will use [pre-trained DenseNet](https://github.com/liuzhuang13/DenseNet) on ImageNet for object classification.

* Configure training options: Training options come with the models as they are initially trained. Depending on the performance of the model on test data, I may tune the model.

* Trainining: Car/Non-car data will be divided into train, test and validation groups. The training data will be used to train Faster R-CNN  for region detection and DenseNet for image classification. 

* Evaluate the trained model: The model will be evaluated on the test and validation data created out of car/non-car dataset. 

* Apply model on the test video: 
  * Input: Create images out of video captured from a car.  Do necessary editing on the images.
  * Run the model: Run the trained model on the images. Output should be region of  car(s) in the image are in rectangle.
  * Output: Create a video by putting toget the frames output of our model.

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
