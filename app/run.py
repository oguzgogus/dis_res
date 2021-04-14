import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///C:\\Users\\ogzpython\\Desktop\\ml\\response_ml\\Disaster_Response_Project\\data\\dis_res.db')
df = pd.read_sql_table('dis_res', engine)

# load model
#model = joblib.load("../models/your_model_name.pkl")
model = joblib.load(r"C:\Users\ogzpython\Desktop\ml\pkls\dis_res\model.pkl")

def tokenize2(text):
 
 
     """input:text
     output: cleaned and tokenized list of the text"""
 
     
     text = text.lower()
     text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
     text = word_tokenize(text) 
     # eliminating stopwords 
     text = [w for w in text if w not in stopwords.words("english")]
     text = [WordNetLemmatizer().lemmatize(w) for w in text]
     
     
     return text
 
def explode_df(df):   

    messages_token = df['message'].apply(tokenize2)
    messages_token = pd.DataFrame(messages_token)
    messages_token = messages_token.explode('message')
    messages_token['message'] = messages_token['message'].astype(str)
    messages_token = messages_token['message'].value_counts()[0:30]
    messages_token = pd.DataFrame(messages_token).reset_index()
    messages_token.columns = ['message','counts']
    
    return messages_token


msg_val_cnts = explode_df(df)







# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message'].reset_index()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals


 
         
    
    
    
    fig = px.bar(genre_counts, x='genre', y='message',title='2) Distribution of Message Genres',template="ggplot2") 
    fig2 = px.scatter(msg_val_cnts,x='message',y = 'counts',size='counts',title='1) Most Frequent Words',template="ggplot2")

    graphs = [fig,fig2
         ]
    
    
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],

    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    
    
    
# genre_counts = df.groupby('genre').count()['message'].reset_index()
# genre_names = list(genre_counts.index)     
# fig = px.bar(genre_counts, x='genre', y='message')     
# fig.show(renderer="png")

