import random
import json
import numpy as np
import pickle

import nltk


from nltk.stem import WordNetLemmatizer
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemitizer = WordNetLemmatizer()

intents = json.loads(
    open('C:/Users/snehp/Desktop/Python Programms/Projects/data.JSON').read())

words = []
classes = []
documents = []
ignore_letter = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append(((words_list), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemitizer.lemmatize(word)
         for word in words if word not in ignore_letter]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('word.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

trainning = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemitizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        if word not in word_pattern:
            bag.append(0)
        else:
            bag.append(1)

    output = list(output_empty)
    output[classes.index(document[1])] = 1
    trainning.append([bag, output])

random.shuffle(trainning)
trainning = np.array(trainning)
train_x = list(trainning[:, 0])
train_y = list(trainning[:, 1])

model = Sequential(
    [
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # < softmax activation here
        Dense(len(train_y[0]), activation='softmax')
    ]
)

model.compile(
    loss=tensorflow.keras.losses.CategoricalCrossentropy(),
    optimizer=SGD(learning_rate=0.01, weight_decay=1e-6,
                  momentum=0.9, nesterov=True),
    metrics=['accuracy']
)

hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')
