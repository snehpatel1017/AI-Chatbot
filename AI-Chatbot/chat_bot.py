import random
import json
import numpy as np
import pickle


import nltk
import streamlit as st
from streamlit_chat import message as st_message

from datetime import datetime


from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemitizer = WordNetLemmatizer()

words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
intents = json.loads(
    open('C:/Users/snehp/Desktop/Python Programms/Projects/data.JSON').read())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemitizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = []
    for w in words:
        if w in sentence_words:
            bag.append(1)
        else:
            bag.append(0)
    return np.array(bag)


def predict(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    result = [[i, r] for i, r in enumerate(res) if r >= 0.9]
    if (len(result) > 0):
        result.sort(key=lambda x: x[1], reverse=True)
        tag = classes[result[0][0]]
        print(result)
        for intent in intents['intents']:
            if intent['tag'] == tag:

                if tag == "time":
                    answer = "{} : {}".format(
                        intent['responses'][0], datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    return answer
                else:
                    return "{}".format(intent['responses'][0])
                break
    else:
        return "Please ask relevent quesion!"


history = []


def generate_answer():
    question = st.session_state.input_text
    answer = predict(question)
    history.append({"message": question, "is_user": True})
    history.append({"message": answer, "is_user": False})
    for chat in history:
        st_message(**chat)


st.title("Hello OpenAI")
st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)
