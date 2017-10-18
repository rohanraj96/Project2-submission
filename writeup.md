#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_imgs/translation.png "Translation"
[image2]: ./report_imgs/shear.png "Shear"
[image3]: ./report_imgs/loss.png "Loss"
[image4]: ./report_imgs/accuracy_and_loss.png "Accuracy and Loss"
[image5]: ./report_imgs/acc_epoch_wise.png "Accuracy per epoch"
[image6]: ./report_imgs/valid_acc.png "Validation Accuracy"
[image7]: ./report_imgs/training_Acc.png "Training Accuracy"
[image8]: ./report_imgs/visualization.png "Visualization"
[image9]: ./street_view/1.png "img1"
[image10]: ./street_view/2.png "img2"
[image11]: ./street_view/3.png "img3"
[image12]: ./street_view/4.png "img4"
[image13]: ./street_view/5.png "img5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rohanraj96/Project2-trafficsignclassifier)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a scatter plot showing the extent of unbalanceness in the classes

![alt_text][image8]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce compute and because my architecture only accepts images with one channel

As a last step, I normalized the image data to "center" the data. This also scales the data within a reasonable range in such a way that during backprop, we do not face a problem of vanishing/exploding gradients due to heavy input scalars.

I decided to generate additional data because of unbalanced classes (as already shown in the figure above). Data augmentation is a very popular technique used to augment our training data from the existing data.

To add more data to the the data set, I used:
    i.) rotation
    ii.) translation
    iii.) brightning
    iv.) flipping
    v.) shearing
    vi.) stretching
    vii.) blurring

Here is are examples of the augmented images:

![alt_text][image1]
![alt_text][image2]

The difference between the original data set and the augmented data set is the 54714 images. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 image									| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x30 				|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x64	|
| RELU					| 												|
| Max Pooling			| 2x2 stride, outputs 5x5x64					|
| FC					|outputs 120									|
| RELU					|												|
| FC					|outputs 84										|
| RELU					|												|
| FC (logits)			|outputs 43										|
   


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet architecture as base and changed the filter numbers in each convolutional layer (increased them to account for a more detailed dataset than MNIST). I also added 2 dropout layers with dropout probability = 0.50 to include regularization in my model.
I also planned to use L2 weight regularization but my model wasn't deep enough so it wasn't exactly overfitting the data as shown in the image:

![alt_text][image7]
![alt_text][image6]

I used tensorboard to draw these graphs and adding L2 regularization would have resulted in underfitting the model so I decided against it. I could have made the model deeper and used 1x1 convolutions but due to limited compute and a lack of financial support, I had to stick with a simple architecture. When I was printing just the validation accuracy, each epoch took roughly 40 minutes. However, when I was logging and printing training accuracy, validation accuracy and the loss, each epoch took close to 15 hours. I wasn't able to figure out why. Please note that all training was done on CPU.

![alt_text][image3]
![alt_text][image4]
![alt_text][image5]

I used Adam optimizer so I also decided against using a learning rate decay as Adam more or less incorporates the essence of learning rate decay in the model.
I used a batch size of 128 because it's more convenient for processors to handle batches in powers of 2. I could not tweak a lot of hyperparameters due to limited compute power so I stuck with 5 epochs and a learning rate of 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ~95
* validation set accuracy of 94.6 
* test set accuracy of 92.05


If a well known architecture was chosen:
* What architecture was chosen? I chose the LeNet architecture as my base as it was small enough to be computed on a CPU.
* Why did you believe it would be relevant to the traffic sign application? I believe it has great application in not just detecting traffic signs but also other obstacles and entities while driving on the road like red lights, pedestrians etc.. We just need the proper dataset to train our models on
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The loss was decreasing and the accuracy was increasing. The model wasn't overfitting the training set as discussed above and in 5 epochs it reached a validation accuracy of 94.6% which is not bad.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt_text][image9] ![alt_text][image10] ![alt_text][image11] 
![alt_text][image12] ![alt_text][image13]

The 3rd and 4th images are particularly under-exposed and hard to classify, whereas the 1st and 5th are relatively easy to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							| Prediction							| 
|:-----------------------------:|:-------------------------------------:| 
| Parking	| Speed limit (60km/h)			| 
| No Entry					| Speed limit (60km/h)							|
| No Stopping		| Speed limit (60km/h)					|
| Right turn ahead						| Speed limit (60km/h)					 			|
| Speed Limit (30 km/h)			| Priority Road					|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This compares in stark contrast to the accuracy on the test set of 12630 test images, which was 92.05%.
However, I suspect this poor performance is due to the vertical lines that have creeped into the image during the preprocessing pipeline. This could be because the training images were of .ppm format whereas these 5 images are of .png format

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 3rd from last code-cell of the Ipython notebook.

After going through the pre-processing pipeline, the images invariably end up with vertical lines throughout the width which occludes most of the image. This might be due to the cv2.cvtColor function.
As a result, the model isn't very confident about any of the predictions it makes and therefore the top softmax probabilities are more or less the same.
