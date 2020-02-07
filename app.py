import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import metrics

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('labelled_dataset.csv', encoding="latin-1")
    df.drop(['Product Name','Brand Name', 'Price'], axis=1, inplace=True)
    df['label']=df['Label']
    X = df['Reviews']
    y = df['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    clf=MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    my_prediction=[]
    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        sentiment_predict=str(my_prediction[0])
        if sentiment_predict=='0':
            sentiment_value='Negative Score'
        elif sentiment_predict=='1':
            sentiment_value='Neutral Score'
        elif sentiment_predict=='2':
            sentiment_value='Positive Score'

    return render_template('index.html',sentiment_analysis = sentiment_value)

if __name__ == '__main__':
    app.run(debug=True)