import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



def load_data(database_filepath):
    
    """ 
   input : database path
   output : X,y to use in ML
   
    """
    
    
    engine= create_engine('sqlite:///{}'.format(database_filepath))
    q = '''select * from dis_res'''
    df = pd.read_sql(q,engine)
    X = df['message']
    # eliminating other columns than categories
    category_names = df.drop(labels=['id','message','original','genre'],axis=1).columns
    Y = df[category_names]
    return X,Y,category_names


def tokenize(text):
    
    """
    input:text
    output: cleaned and tokenized list of the text
    """
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    text = word_tokenize(text) 
    # eliminating stopwords 
    text = [w for w in text if w not in stopwords.words("english")]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    
    return text


def build_model():
    
    """
    output: model
    
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    #my computer thakes enourmous time to perform the task with more paramaters.
    #a lot of params can be added such as:
    # parameters = {
    #     'clf__estimator__criterion':['gini','entropy'],  
    #     'clf__estimator__min_samples_split':[10,110],
    #     'clf__estimator__max_depth':[None,100,500]
    #           }
    

    parameters = {
        'clf__estimator__n_estimators': [50]
        }
    
    
    model = GridSearchCV(pipeline, param_grid=parameters)


    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    output: prints classification report 
    """
    
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test,Y_pred,target_names = category_names)
    print(report)
    return report



def save_model(model, model_filepath):
    
    pickle.dump(model,open(model_filepath,'wb'))

    pass


def main():
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