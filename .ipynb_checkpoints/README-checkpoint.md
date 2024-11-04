
                    # Capstone Proposal - Inventory Monitoring at Distribution Centre
                                             #Betty R Mtengwa
                                            # 4 September 2024

## Background
This capstone project proposal is part of the fulfilment of AWS Machine Learning Engineer Nanodegree with Udacity. The aim of the project is to serve as a demonstration of how an end-to-end machine learning model would look like. A traditional way of managing inventory has proved to have several challenges as the systems often struggle to adapt quickly to changes in demand and the system provide insufficient information on real-time stock levels (StoreFeeder, 2024). As the business is not able to track its stock and information is not up to date, this can result in loss of 
sales. In addressing these challenges businesses have turned to distributed inventory management which is a modern way of managing inventory stock. Amazon distribution centres have adopted the modern way of managing inventory and this helps to track products from obtaining the stock to customer shipment. One of the modern ways they have adopted is by using robots to move objects as part of their operations in a dayto-day business. However, they is need to check how many items have been placed in each bin to track inventory and making sure that delivery consignments have the 
correct number of items. This is carried out using a pre-trained convolutional neural network machine learning model. Amazon Bin Image Dataset is a dataset that contains images of bins showing items that have been placed by the robot. The data is publicly accessible. The dataset is large, and it is time consuming to physically assess each picture, hence we are going to train a machine learning model and this also saves time.To complete this project, I used AWS SageMaker to train and test the model. I also used python 3x as the programming language.

## Problem Statement
The aim of the project is to build a model that can count the number of objects that are placed in each bin by the robots that helps to move objects around the factory.The model should be able to accurately predict the number of objects in a bin and also identify false prediction.


## Dataset Description
The Amazon Bin Image dataset contains almost over 500, 000 images of bins in a pod in an operating Amazon Fulfilment Centre. Amazon Fulfilment Centres are known for delivering millions of products to customers worldwide with the help of robots and computer vision technologies. The images were captured when the robots where in operation. The dataset is obtained from Amazon s3 bucket and 
uploaded to Amazon s3 container (this is where the input data will be stored). As the dataset is quite large, a sample of 10, 000 images will be used. The images are resized to match with pre-requisites required when using the pre-trained model. The data is split into three portions 80% for training, 10% for testing and the remaining 10% for validation. The main reason for splitting the data is that the training set is used to train our model to accurately count the number of items that have been placed in each bin, test set is used to evaluate the performance of the 
model, how well the model has performed and the validation set is used to fine-tune hyperparameters based on the performance of the model.

## Benchmark Model
As a baseline model l intend to use a pre-trained convolutional neural network model obtained from AWS cloud. This is a simple classification network for counting tasks. I will also use resnet18 for image classification. Resnet accepts images with a size of 224 * 224 and the images will be resized during the preprocessing stage.

## Evaluation Metrics
The quality of the model will be evaluated using a standard metrics. Confusion matrix and cross entropy loss will be used. To achieve best results during training, hyper parameter optimisation may possibly be needed to ensure the training process will show expected results.

## An outline of the project
The following outlines the step=by-step process that will be carried out in this project;
1. Data collection
• Fetching data from Amazon S3 bucket and uploading to s3 container
2. Data Preparation
• Pre-process the data and if necessary removing irrelevant features 
from the data
• Reducing the size of the dataset
• Divide the data into train, test and validate
3. Model Development
• Finetuning the hyperparameters of the model
• Training the model, getting results and observe its performance to see 
if they are any anomalies
• Record the evaluation metrics
4. Model Deployment
• The first step is to verify that the model is performing as expected
• Deploy the model into productio