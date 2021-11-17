# Disaster Response Pipeline Project

### Project Overview:
This project is completed as part of Udacity Data Science Nanodegree program. The aim of the project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### File Description:
1. ETL Pipeline
process_data.py: The data cleaning pipeline that loads the messages and categories datasets, merges the two datasets, cleans the data, sStores it in a SQLite database
2. ML Pipeline
train_classifier.py: The machine learning pipeline that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file
3. Flask Web App
Data visualizations using Plotly in the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
