import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    """
    Importing and merging datasets

    Inputs:
        messages_path -> Path to the CSV file containing messages
        categories_path -> Path to the CSV file containing categories
    Output:
        data -> Combined data containing messages and categories
    """

    # load messages dataset
    messages = pd.read_csv(messages_path)
    # load categories dataset
    categories = pd.read_csv(categories_path)

    # merge datasets
    data = pd.merge(messages, categories)

    return data

def clean_data(data):
        # create a dataframe of the 36 individual category columns
        categories = data['categories'].str.split(';', expand=True)

        # select the first row of the categories dataframe
        row_one = categories.iloc[0]

        # use this row to extract a list of new column names for categories.
        # one way is to apply a lambda function that takes everything
        # up to the second to last character of each string with slicing
        category_colnames = row_one.apply(lambda x: x[:-2]).values.tolist()

        # rename the columns of `categories`
        categories.columns = category_colnames

        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].str[-1]
            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])

        # drop the original categories column from `data`
        data.drop(['categories'], axis=1, inplace=True)

        # concatenate the original dataframe with the new `categories` dataframe
        data = pd.concat([data, categories], axis=1)

        # drop duplicates
        data.drop_duplicates(inplace=True)

        # Remove rows with a value of 2 in 'related' column to make it binary classification
        data = data[data['related'] != 2]

        return data


def save_data(data, database_filename):
    """
    Save data into SQLite database

    Arguments:
        data -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """

    engine = create_engine('sqlite:///' + database_filename)
    data.to_sql('DisasterResponseMaster', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_path, categories_path, database_path = sys.argv[1:]

        print('Loading data...\n')
        data = load_data(messages_path, categories_path)

        print('Cleaning data...')
        data = clean_data(data)

        print('Saving data...\n')
        save_data(data, database_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the paths of the messages and categories ' \
            'datasets as the first and second argument respectively, as ' \
            'well as the path of the database to save the cleaned data ' \
            'to as the third argument. \n\nExample: python process_data.py ' \
            'disaster_messages.csv disaster_categories.csv ' \
            'DisasterResponse.db')

if __name__ == '__main__':
    main()