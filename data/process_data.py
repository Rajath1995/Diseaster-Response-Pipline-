# import libraries
import sys
import pandas as pd
import numpy as np
import itertools
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load data
    
    Arguments:
        messages_filepath - Path to  file containing messages
        categories_filepath - Path to  file containing categories
    Output:
        df - Combined data containing messages and categories
    """
    messages=pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath, dtype=str)
    df = messages.merge(categories, how='outer',on=['id'])
    
    return df

def clean_data(df):
    
    """
    Clean Data function
    
    Arguments:
        df - raw data Pandas DataFrame
    Outputs:
        df - clean data Pandas DataFrame
    """
    categories = df['categories'].str.split(';',expand=True)
    row=np.asarray(categories.head(1)).flatten()
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df=df.drop(columns=['categories'],axis=1)
    df = pd.concat([df, categories],axis=1)
    df.drop_duplicates(keep=False, inplace=True)
    return df


def save_data(df, database_filename):
    
    """
    Save Data function
    
    Arguments:
        df - Clean data
        database_filename - database file
    """
    
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('merged_dataset', engine, index=False,if_exists='replace')     
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()