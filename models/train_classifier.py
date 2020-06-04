"""
TRAIN CLASSIFIER

Disaster Resoponse Project

How to run this script
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Arguments:

    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
    
"""
# import libraries
import sys
import sqlite3
from sqlalchemy import create_engine
import pandas as pd 
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    
    """
    Load Data 
    
    Arguments:
        database_filepath
        
    Output:
        X -> feature DataFrame
        
        Y -> label DataFrame
        
        category_names
        
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('merged_dataset',engine)
    df = df.drop(['child_alone'],axis=1)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text,url_place_holder="urlplaceholder"):
    """
    Tokenize function
    
    Arguments:
    
        text - list of text messages.
        
    Output:
        clean_text - tokenized text
    """
    
    regex_url= 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    find_url=re.findall(regex_url,text)
    for url in find_url:
        text=text.replace(url,url_place_holder)
    tokenization=word_tokenize(text)
    lemmatization=WordNetLemmatizer()
    final_text=[]
    for word in tokenization:
        word=lemmatization.lemmatize(word).lower().strip()
        final_text.append(word)
    return final_text


def build_model():
    
    """
    Build Model
    
    The  Scikit Learn ML Pipeline that process text messages
    with help of NLP and apply to the classifier.
    
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(SVC())),
    ])
    parameters_grid = {'classifier__estimator__C':[1,10,100], 'classifier__estimator__kernel':['linear','rbf']}
    
    Cj = GridSearchCV(pipeline,param_grid=parameters_grid)

    return Cj


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    evaluate_model
    
    This is a performance metric output function
    
    Output:
        f1score , accuracy
    """
    y_pred = model.predict(X_test)
    Y_test=np.asarray(Y_test)
    # Print classification report on test data
    for column in range(0,Y_test.shape[1]):
        print(classification_report(Y_test[:,column],y_pred[:,column],output_dict=True))
    # Print classification report on test data
    accuracy_pip2=[]
    for column in range(0,Y_test.shape[1]):
        accuracy_pip2.append(accuracy_score(Y_test[:,column],y_pred[:,column]))
    print("The overall accuracy of the model is found to be:", np.mean(accuracy_pip2))
    pass

def save_model(model, model_filepath):
    
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline Scikit Pipelin object
        pickle_filepath - destination path 
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass
    


def main():
    """
    Main function
    
    1. it extracts data from sql db
    2. Calls the ML model function
    3. Model performance
    4. It saves model into pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()