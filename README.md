# Detecting Synthetic Images in AWS SageMaker

In this project I attempted to use the [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data) dataset from kaggle for Computer Vision classification.

This project serves as the capstone project for the AWS Machine Learning Nanodegree by Udacity. In this project I leveraged the usage of AWS SageMaker for preprocessing the data, tuning hyperparameters, training a model, and deploying the model as an endpoint for inference. I also provide a Step Functions workflow that allows the triggering of a Lambda Function to predict whether an image uploaded to an S3 bucket is real or fake.

This project is motivated by the raise of Generative AI, and the rapid improvement of its capabilities in image generation. This presents huge opportunities and technological advancements, however, worries about the ability of automatic systems (or even people) to differentiate real from fake images have also been growing.

## Project Setup
The project consists of the following files:
- `proposal/`: folder that contains the markdown and pdf of the proposal for the project.
- `src/`: folder that contains all source data for the project.
    - `train_and_deploy.ipynb`: this notebook contains all the step for dataset analysis, hyperparameter tuning, training of the final model, and deployment of the endpoint.
    - `hpo.py`: this Python code contains the creation, training and testing of a fined-tuned VGG neural network, using hyperparameters given from the arguments.
    - `train_model.py`: this Python code contains the creation, training and testing of a fined-tuned VGG neural network, using hyperparameters given from the arguments, and adding a hook for the Sagemaker Profiler and Debugger. 
    - `requirements.txt`: this file contains the requirements for the hpo and train_model Python files. This is required since the notebook launches the training jobs in a new machine which doesn't have all requirements installed. This file tells Sagemaker which Python libraries to install.
    - `lambdas/`: folder that contains the source code for the lambda functions, one folder for each function, and the requirements and Dockerfile for the imageClassifier.
        - `execution-detail.json`: definition of the Step Functions workflow.
- `README.md`: a basic documentation of this project.
- `report.md`/`report.pdf`: a comprehensive report of this project.

To create and train the model in this project, we have to execute all the steps in the `train_and_deploy` notebook.

## Libraries/Frameworks used

- [Black formatter](https://github.com/psf/black): PEP8 formatter that helps the project code keep on track with standards.
- [Kaggle](https://www.kaggle.com/): Website that hosts the dataset used for this model.
- [Python3.10](https://www.python.org/downloads/): Latest version of the Python language available in Sagemaker Notebook kernels.
