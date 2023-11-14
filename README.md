# Income Prediction with Adult Census Data

## Overview

This project focuses on predicting an individual's income based on various demographic and socioeconomic features from the U.S. Census Bureau's dataset. The dataset contains information such as age, education, occupation, marital status, and more, and we aim to build a machine learning model that can accurately predict whether a person's income exceeds $50,000 per year.

## Dataset

The dataset used in this project is the "Adult Census Income" dataset. It contains the following features:

1. **Age**: The age of the individual.
2. **Workclass**: The type of employment, such as private, self-employed, government, etc.
3. **FNLWGT**: The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian noninstitutional population of the US.
4. **Education**: The highest level of education achieved.
5. **Education-Num**: A numerical representation of education, often representing the number of years of education completed.
6. **Marital Status**: Whether the individual is married, divorced, widowed, etc.
7. **Occupation**: The type of occupation the individual is engaged in.
8. **Relationship**: The role in the family (e.g., husband, wife, etc.).
9. **Race**: The racial group the individual belongs to.
10. **Sex**: The gender of the individual.
11. **Capital Gain**: The capital gains made by the individual in the past year.
12. **Capital Loss**: The capital losses incurred by the individual in the past year.
13. **Hours per week**: The number of hours the individual works per week.
14. **Native-country**: The age of the individual.
14. **Income**: The target variable, indicating whether the individual's income exceeds $50,000 (1) or not (0).

## Project Structure

The project is organized into the following directories:

- **app_tracking**: Contains logging scripts and exception handling scripts.
- **artifacts**: Contains all data files, models, and logs file.
- **notebooks**: Jupyter notebooks for data exploration, data preprocessing, model training, and evaluation.
- **PipelinePrediction**: Contains Python scripts for prediction.
- **Pipeline_training**: Contains source code files, including scripts for data ingestion, data validation, data preprocessing, model training, and model evaluation.
- **Utils**: Contains utility files.
- **create_project_structure.sh**: A file to create the general project structure.
- **application**: A file where the code flows.
- **requirements.txt**: A file specifying the required Python packages and their versions for running this project.
- **setup.py**: A file used for packaging and distribution purposes.
- **test.py**: A file used for testing the train and predict pipeline.

### Classification Models Used:

- **GradientBoostingClassifier**
- **HistGradientBoostingClassifier**

### Preprocessing Models Used:

- **StandardScaler**
- **OneHotEncoder**

### Tools Used:

- **Git** - Version Control
- **Docker** - Containerization
- **AWS** - Deployment

### Libraries Used:

- **pandas**
- **numpy**
- **matplotlib**
- **Flask**
- **Flask-Cors**
- **scikit-learn**
- **scipy**
- **joblib**
- **cassandra-driver**

### AWS CD Deployment with AWS Beanstalk

1. Login to AWS console.
2. Create an application.
3. Create a code pipeline.
4. Domain link - [incomeprediction.ap-south-1.elasticbeanstalk.com](http://incomeprediction.ap-south-1.elasticbeanstalk.com) Not working

## Installation

To run this project locally, you need to create a Python 3.9 virtual environment and install the required packages by running the following commands:

```bash
conda create -p venv python=3.9 -y
source activate ./venv
pip install -r requirements.txt
python application.py
