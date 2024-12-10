import streamlit as st
import dill
import nltk
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string
punctuations = string.punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

vectorizer = dill.load(open('artifacts/feature_engineering.pkl', 'rb'))
model = dill.load(open('artifacts/model_trainer.pkl', 'rb'))

st.title('SMS/Spam Classifier')

input_sms = st.text_area("Enter the message")


def transform_text(text):
    # Convert into lower case
    text = text.lower()
    #tokenisation
    text = nltk.word_tokenize(text=text)
    
    #Remove special characters
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    
    text = y[:]
    y.clear()

    # Remove stop words and punctuations
    for i in text:
        if i not in stop_words and i not in punctuations:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# 1. Preprocessing
transformed_sms = transform_text(input_sms)

# 2. vectorise
vectorised_sms = vectorizer.transform([transformed_sms])

# 3. predict
submit = st.button('Predict')
if submit:
    prediction = model.predict(vectorised_sms.toarray())
    print(prediction)

    if prediction == [1]:
        st.write("SPAM mesasge")
    else:
        st.write("NORMAL message") 

