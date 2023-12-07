# import libraries
import sys
import numpy as np
import pandas as pd
import nltk
from sqlalchemy import create_engine

# download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])
# import statements
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import ML modules
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_path):
    """
    Load data from database

    Arguments:
        database_path -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_path)
    df = pd.read_sql_table('DisasterResponseMaster', engine)
    x = df.message
    y = df.iloc[:, 4:]

    # listing the columns
    category_name = list(np.array(y.columns))

    return x, y, category_name


def tokenize(text):
    """
    Tokenize the text

    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    # url regular expression and english stop words
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    words = stopwords.words('english')

    # get list of all urls using regex
    de_urls = re.findall(regex, text)

    # replace each url in text string with placeholder
    for i in de_urls:
        text = text.replace(i, 'urlplaceholder')

    # remove punctuation characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # tokenize text
    token = word_tokenize(text)

    # initiate word_net_lemmatizer
    word_net_lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean = []
    for i in token:
        if i not in words:
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = word_net_lemmatizer.lemmatize(i).lower().strip()
            clean.append(clean_tok)

    return clean


def build_model():
    """
    Build pipeline for message classification, the parameters for the pipeline are obtained from GridSearchCV

    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4, cv=2)

    return cv


def get_scores(y_test, y_pred):
    """
    Function to calculate the F1 score, precision and recall for each category of the data
    This is a performance metric of my own creation.

    Arguments:
        y_test -> Actual labels on test samples
        y_pred -> Predicted labels on test samples

    Output:
        scores -> Pandas frame with the precision and recall on each category
    """
    scores = pd.DataFrame(columns=['category', 'precision', 'recall', 'f1_score'])
    for j, cat in enumerate(y_test.columns):
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:, j],
                                                                               average='weighted')
        scores.set_value(j + 1, 'precision', precision)
        scores.set_value(j + 1, 'category', cat)
        scores.set_value(j + 1, 'recall', recall)
        scores.set_value(j + 1, 'f1_score', f1_score)
    print('precision:', scores['precision'].mean())
    print('recall:', scores['recall'].mean())
    return scores


def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluate the classificaiton model

    This function applies a ML pipeline to a test set and prints out the model performance (accuracy, F1 score, precision and recall)

    Arguments:
        model -> A valid scikit ML Pipeline
        x_test -> Test features
        y_test -> Test labels
        category_names -> label names (multi-output)
    """
    y_pred = model.predict(x_test)
    # print the overall accuracy of the model
    accuracy = (y_pred == y_test).values.mean()
    print('Model Overall Accuracy: {}'.format(accuracy))

    # print the F1 score, preision and recall for each category
    print(get_scores(y_test, y_pred))


def save_model(model, model_path):
    """
    Save the classification model

    This function saves trained model as Pickle file, to be loaded later.

    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        model_path -> destination path to save .pkl file

    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n ')
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(x_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()