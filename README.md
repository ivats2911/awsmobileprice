# MLAWS-SagemakerProject
Mobile Price Classification using SKLearn in AWS SageMaker
This project demonstrates how to build, train, and deploy a machine learning model for mobile price classification using a custom script with scikit-learn (SKLearn) on AWS SageMaker. The workflow includes data preprocessing, model training, and deployment, leveraging the managed services provided by SageMaker for a scalable machine learning pipeline.

Project Overview
The objective of this project is to classify mobile phones into different price ranges based on various features. The project follows these major steps:
1.	Data Preparation: Loading and preprocessing the dataset.
2.	Model Training: Training a RandomForest model using a custom SKLearn script.
3.	Model Deployment: Deploying the trained model to a SageMaker endpoint for real-time predictions.
4.	Model Inference: Making predictions using the deployed endpoint.
5.	Cleanup: Deleting the endpoint to avoid unnecessary costs.
Prerequisites
Before starting this project, ensure you have:
•	An AWS account with SageMaker permissions.
•	AWS CLI configured on your machine with appropriate permissions.
•	Python installed with Boto3 and SageMaker SDK.
Setup and Configuration
1.	Set Up AWS SageMaker Environment:
o	Create an S3 bucket to store your data and model artifacts.
o	Open SageMaker in the AWS Management Console.
o	Set up a SageMaker notebook instance with necessary permissions.
2.	Install Required Python Packages:
o	Make sure to have boto3, sagemaker, and pandas installed in your Python environment.
#########
pip install boto3 sagemaker pandas
#########
Data Preparation

1.	Load Dataset:
o	Read the CSV file containing mobile data into a pandas DataFrame.
o	Example:
df = pd.read_csv("mob_price_classification_train.csv")

2.	Data Exploration and Preprocessing:
o	Check for null values and data distribution.
o	Split the dataset into training and testing sets.
#######
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
3.	Upload Data to S3:
o	Save the processed data to CSV files and upload them to your S3 bucket for SageMaker to access.
python
Copy code
sess.upload_data(path="train-V-1.csv", bucket=bucket, key_prefix=sk_prefix)
Training the Model
1.	Create Training Script (script.py):
o	Implement the training logic using SKLearn’s RandomForestClassifier.
o	The script reads input data, trains the model, and saves it using joblib.
2.	Configure SageMaker Estimator:
o	Define a SKLearn estimator in SageMaker, specifying the training script and parameters.
python
Copy code
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    hyperparameters={"n_estimators": 100, "random_state": 0},
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)
3.	Launch Training Job:
o	Start the training job and monitor the process in the SageMaker console.
python
Copy code
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True)
Deploying the Model
1.	Create and Deploy Model:
o	Use the trained model artifacts to create a SKLearnModel object.
o	Deploy the model to a SageMaker endpoint.
python
Copy code
model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)
Making Predictions
1.	Predict on New Data:
o	Use the deployed endpoint to make predictions.
python
Copy code
predictor.predict(testX[features][0:2].values.tolist())
Cleaning Up
1.	Delete the Endpoint:
o	To avoid incurring costs for unused resources, delete the endpoint once you are done.
python
Copy code
sm_boto3.delete_endpoint(EndpointName=endpoint_name)



Conclusion
This project provides an end-to-end example of how to use AWS SageMaker for machine learning tasks, from data preparation and model training to deployment and inference. By leveraging AWS services, you can build scalable machine learning solutions without managing the underlying infrastructure.
