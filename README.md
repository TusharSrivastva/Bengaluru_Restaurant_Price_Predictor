# Bengaluru Restaurant Price Predictor

## Problem Statement
The **aim** of this project is to **predict** the **cost** of dining for two people at a **restaurant** from a dataset of restaurant **data** extracted from **Zomato** website via web scrapping.

## Approach
1. Data Ingestion
	* Data is added to the project from a CSV file
	* Data is split into test and train Datasets
2. Data Transformation
	* Train data is fit and transformed using SKLearn ColumnTransformer and Pipelines
	* Test data is transformed
3. Model Training
	* Multiple models are trained on test dataset
	* Best model performing is selected
4. Training Pipeline
	* A training pipeline is created to generate preprocessor and model pickle files
5. Predict Pipeline
	* A predict pipeline is created to take input from flask app and generate output predictions.
6. Flask App
	* A flask app is created to take input from end user and generate predictions
7. AWS Deployment
	* Flask app is deployed to AWS Beanstalk
8. CI/CD pipeline
	* A CI/CD pipeline is setup using Jenkins

## About the data
The [dataset](https://www.kaggle.com/datasets/rishikeshkonapure/zomato) used in this project is taken from Kaggle. It contains data of the **restaurants** located in **Bengaluru**. The dataset contains **17 columns** and **51717 rows**. The columns are as follows:

-   `url` - The Zomato URL of the restaurant.
-   `address` - The address of the restaurant.
-   `name` - The name of the restaurant.
-   `online_order` - Whether restaurant take online order or not
-   `book_table` - Whether restaurant take online bookings of tables
-   `rate` - Zomato rating of the restaurant out of 5.0.
-   `votes` - Likes received by the restaurant.
-   `phone` - Phone number of the restaurant.
-   `location` - Area where the restaurant is located
-   `rest_type` - Type tags of restaurant (Cafe, Casual Dinning, etc)
-   `dish_liked` - Dishes liked by the reviewers.
-   `cuisines` - Cuisine of the restaurant.
-   `reviews_list` - List of reviews left by the reviewers.
-   `menu_item` - List of items on the menu.
-   `listed_in(type)` - Type of restaurant (Buffet, bar, etc.).
-   `listed_in(city)` - Area where the restaurant is located.

Target variable:

-   `approx_cost(for two people)` - Cost of dining for two people.

## Screenshots of the App
* ![Local 1](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/Screenshots/Local%201.png)
* ![Local 2](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/Screenshots/Local%202.png)


## Screenshot of AWS Deployment
* ![Elastic Beanstalk](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/Screenshots/Elastic%20Beanstalk.png)


## Screenshots of Jenkins Pipeline
* ![Jenkins 1](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/Screenshots/Jenkins%201.png)
* ![Jenkins 2](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/Screenshots/Jenkins%202.png)


## Exploratory Data Analysis Notebook
* [EDA](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/notebook/Zomato_EDA.ipynb)

## Model Training Notebook
* [Model Training](https://github.com/TusharSrivastva/Bengaluru_Restaurant_Price_Predictor/blob/main/notebook/Zomato_Model_Training.ipynb)