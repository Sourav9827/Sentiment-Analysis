import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import nltk, re, string
from nltk.corpus import stopwords, twitter_samples
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from Utilities import process_tweet
from nltk.corpus import stopwords

app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

logistic=pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('cv.pkl','rb'))

def predict_sentiment(tweet):
    tweet = process_tweet(tweet)
    tweet = cv.transform([tweet])
    if logistic.predict(tweet) == 1:
        sentiment = 'Positive Sentiment'
    elif logistic.predict(tweet) == 0:
        sentiment = 'Negetive Sentiment'
    else:
        sentiment = 'Neutral Sentiment'
    return sentiment

@app.route('/',methods=['GET','POST'])
def index():
    
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():

    sentiment=predict_sentiment(request.form['tweet'])
    print(sentiment)
    return render_template('prediction.html', prediction_text="Your tweet is  of {}".format(sentiment))

if __name__=="__main__":
    app.run(debug=True)

