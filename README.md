
Project3:                           
Classification
================
Manish Reddy Challamala,
November 22, 2018 ,manishre@buffalo.edu

For detailed explaination please visit the below link:

[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)

## Abstract
To train the MNIST Dataset with different classifiers and combine the
result of individual classifier to make a final decision.

## 1 Introduction

The goal of this project is to implement different classification methods for
MNIST data set(handwritten digit images) and combine the results of individual
classifier to make a final decision.
The features are obtained from two different source:

1. MNIST Dataset
2. USPS Dataset

The above feature data sets are trained by using following methods:

1. Logistic Regression using Softmax.
2. Dense Neural networks[DNN].
3. Convoluted Neural netwotk [CNN].
4. Random Forest
5. Support Vector Machine [SVM]

Train the models using MNIST Dataset and test it on both MNIST and USPS
Dataset.


## 2 Theory

## 3 Logistic Regression Model with softmax function

The logistic regression model is a classification model with linear regression
algorithm which tries to predict y for a given x.

1. The logistic regression gives the probability to which output each input
    belongs.
2. To generate probability for each input logistic regression takes a function
    that maps the output between the range of 0 and 1 for all values of input
    x but as our data consists of multiple classes [i.e; from 0 to 9 classes] we
    convert our binary classification into multiclass classification problem.
4. The hypothesis for logistic regression 
5. The above equation is called the logistic function or the sigmoid function.
6. we calculate the loss function by taking the derivative of logistic function.
7. we calculate the maximum likelihood of the data by minimizing the loss
   function either by increasing or decreasing the weights that best fit for our data.
   we can achieve this by taking the partial derivative of loss function.
   Which is nothing but using the SGD and update the weights iteratively.
   
## 4 Neural Network

The neural network model is a classification model which tries to predict output
y for a given input x

1. The neural network model contain two phase:
    1 Learning Phase
    2. Prediction Phase
2. In learning phase, the neural network takes inputs and corresponding out-
    puts.
3. The neural network process the data and calculates the optimal weights
    to gain the training experience.
4. In prediction process the neural network is presented with the unseen data
    where it predicts a output by using its past training experience on that
    given data.
5. A neural network model can be build by using different layers:
    1. Dense Layer : A linear Operation in which every input is connected to
    every output by weights
    2. Convolution Layer: A linear operation in which input are connected to
    output by using subsets of weights of dense layer.

## 5 Random Forest

1. Random forest is a supervised Algorithm, Which creates the decision tress
    on randomly selected data samples.
2. From each decision tress the random forest algorithm gets the individual
    prediction values.
3. The Final prediction values are the values with most votes in individual
    prediction results.
4. Random Forest is known for its high accuracy and it does not suffer over-
    fitting problem because it takes the average of all predictions.
<img src="https://github.com/manish216/Project-3/raw/master/output/randomforest.JPG" width=500/>
                                    fig[1] :Block Diagram of Random Forest
     
## 6 Support Vector Machine

1. SVM is a supervised learning classification algorithm.
2. SVM differs from all other classifications algorithms,Basically Machine
    learning algorithms tries to find a boundary that divides the data where
    error is minimized.
3. But the SVM chooses a decision boundary that maximizes the distance
    from nearest data points from all classes. which is the optimal decision
    boundary.

## 7 Confusion Matrix

1. Confusion Matrix is said to be the performance measurement of machine
    learning classification problem.
2. It is help-full in measuring the precision, accuracy, specidicity.
3. The Confusion Matrix will give 4 possible type of values
    1. True Postive: The predicted value is positive and its true. which are
    plotted in diagonal elements of matrix.
    2. True Negitive: The predicted negitive and its true.
    for example we predicted that digit 8 is not 0 and its true.
    3.False Positive: The predicted is positive and its false.which are plotted
    in upper triangular elements of matrix
    for example we predicted digit 5 has 6 but it is 5.
    4.False Negitive: The predicted is negitive and its false.which are plotted
    on lower triangular elements of matrix.
    for example we predicted digit 5 has 6 but it is 6.


## 8 Experimental Setup:

The experimental setup consists of three steps:

1. Data Pre-processing
2. Logistic regression using Softmax Activation Function
3. Dense Neural Network
4. Convolutional Neural Network
5. Random Forest
6. Support Vector Machine

### 8.1 Data Pre-Processing:

1. In data pre-processing, we have 2 data sets: 1. MNIST Dataset and 2.
    USPS dataset.
2. Firstly, we are splitting the data into training data, validation data and
    testing data for given two data set.
3. Total we will be having 2 feature sets generated from the 2 given data sets,
    Where we split the raw data in accordance to our program requirement.
4. In this process, we are reading the raw image data from a ’mnist.pkl.gz’
    file and splitting the data into training, validation and testing data.
5. For USPS Dataset, wreading the images from the Numerals folder and
    converting the images into matrix and normalizing the images by iterating
    on each image present in that folder.
6. Dimensions for each Dataset is given Below:
    where rows = no of samples and columns = no of features.

### (a) MNIST Dataset
1. Feature Matrix: 70000 X 784
2. Target Vector : 70000 X 1
3. Training Feature Matrix: 50000 X 784
4. Validation Feature Matrix: 10000 X 784
5. Testing Feature matrix: 10000 X 784

### (b) USPS Dataset
1. As USPS Dataset is only used for testing we are not splitting thedata.
2. Feature Matrix: 19999 X 784
3. Target Vector : 19999 X 1


### 8.2 Logistic Regression using Softmax activation function:

1. In logistic Regression, initialize the random weights.
2. Calculating the optimal weights by plugging the input features and the
    weight matrix to the softmax function equation[1]
3. By using SGD, we are calculating the loss function at each iteration and
    updating weights by using equations [4].
4. After, getting the optimized weights we are predicting the outputs for the
    new unseen data and compare the predicted output with the actual output
    which gives our accuracy.

#### 8.2.1 Experimental Results:

For detailed explaination of strength and weakness of confusion matrix please visit the below link:
[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)

### 8.3 Neural network:

1. In this project I have implemented both Dense and Convoluted neural
    network.
2. For Dense neural network ,Creating a model with 3 layers, 1. input layer
    2. hidden layer 3. output layer
3. No of nodes for each layer is given below:
    1.No of nodes in input layer = No of features in the data set
    2. No of nodes in hidden layer 800
    3.No of nodes in output layer 10 [because we have 10 digits to classify.]


4. Activation functions used in hidden layer is relu [rectified linear unit] be-
    cause it introduces the non linearity in the network and softmax function
    is used on the output layer to predict the target class.
5. For Convolution neural network, Creating a model with 4 layers, 1. input
    layer 2. 2 hidden layers 3. output layer.
6. No of nodes in each layer are given below:
    1. No of input nodes = No of features
    2. No of nodes in hidden layer 1 is 64 and no of nodes in hidden layer 2 is 32
    3. The output layer is dense layer which has 10 nodes.
7. By using the above parameters we create 2 models.
8. we run these models by plugging in the appropriate data to it and train the model.
9. After training the model we test the model using the unseen data to predict
    that which class output belongs with given two datasets.

#### 8.3.1 experimental Results:
For detailed explaination of strength and weakness of confusion matrix please visit the below link:
[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)

### 8.4 Random Forest

1. Using the sklearn library and creating the model with nestimators =100,
    remaining all features are default.

#### 8.4.1 Experimental result:

For detailed explaination of strength and weakness of confusion matrix please visit the below link:
[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)

### 8.5 Support Vector Machine

1. Using the Sklearn library and creating 3 models with following parameters
    1.kernel = linear
    2. kernel =’rbf’(radial function) with gamma =
    3.kernel=’rbf’and remaining default parameters.

#### 8.5.1 Experimental Results:

For detailed explaination of strength and weakness of confusion matrix please visit the below link:
[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)

### 8.6 Ensemble Learning with Hard Voting

1. Make a single model by combining the 4 models.
2. predict the values of the four models, The dimension of the combined
    predicted values is (10000,1).
3. To find the accuracy and confusion matrix,compare the predicted values
    with the actual testing target values.

#### 8.6.1 Experimental Results

For detailed explaination of strength and weakness of confusion matrix please visit the below link:
[link for report pdf](https://github.com/manish216/Project-3/blob/master/proj3.pdf)






