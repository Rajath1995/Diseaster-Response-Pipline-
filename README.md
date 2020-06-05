# Diseaster-Response-Pipline

<div align="center">
    <img src="/images/img1.PNG".png" width="400px"</img> 
</div>

# Description

The message dataset contains pre labeled messages from real life disaster events. This Project builds a Naural Language Processing (NLP) model to categorize the messages.

Link to code :- https://github.com/Rajath1995/Diseaster-Response-Pipline-

The project is completed in several steps:

Step 1:- **Data Processing**

This step involves building an ETL pipline to process data, clean the data and save them into a sql db.

Step 2:- **ML Model**

This steps builds a machine learning model utlizing Natural language processing and Multioutput classifer.

Step 3:- **Building the webapp**

This steps utilizes Flask to host the application on the web.

The necessary packages to run to run this application (Python)

1. SQLLITE
2. SQLALCHEMY
3. PANDAS
4. NUMPY
5. NLTK
6. SKLEARN
7. NLTK PUNKT, WORDNET AND
8. ITERTOOLS.

# Executing the project:

Step1 : ETL script to process data and save the data into database db:-

**python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**

Step2 : Run the Machine Learning script to build the model and save the classifier as pickle file:-

***python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl***

Step3 : Bring up the web application:-

***python run.py***

and check the below link for the web applicaiton to be displayed.

http://localhost:3001/

Authors 
Rajath Nagaraj
Masters in Data Science.

***Acknowledgements***

1. Figure Eight for providing the relevant dataset to train the model
