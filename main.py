

#https://github.com/SARIT42/EnglishToFrench-Translator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow
from keras.utils.data_utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Embedding,Dense, LSTM, Bidirectional,RepeatVector, GRU, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import SparseCategoricalCrossentropy

import re
from string import punctuation
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

df = pd.read_csv('/EnglishToFrench-Translator/new.csv')

#Separating English and French languages.
eng = df['English words/sentences']
fra = df['French words/sentences']


# Cleaning the text of Punctuations and Unnecessary characters/numbers
def clean(string):
    string = string.replace("\u202f", " ")
    string = string.lower()

    for p in punctuation + "«»" + "0123456789":
        string = string.replace(p, " ")

    string = re.sub('\s+', ' ', string)
    return string

eng = eng.apply(lambda x:clean(x))
fra = fra.apply(lambda x:clean(x))


#finding the length of the texts in English and French texts
def word_count(line):
    return len(line.split())

df['English_word_count'] = df['English words/sentences'].apply(lambda x: word_count(x))
df['French_word_count'] = df['French words/sentences'].apply(lambda x: word_count(x))


#TEXT PREPROCESSING FUNCTIONS FOR MODEL TRAINING

#Tokenizing Text
def create_tokenizer(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer


#Finding the maximum sentence length of a language text
def max_sentence_length(lines):
    return max(len(sentence.split()) for sentence in lines)


#Token sequencing and Padding
def encode_sequences(tokenizer,sentences,max_sent_len):
    text_to_seq = tokenizer.texts_to_sequences(sentences)
    text_pad_sequences = pad_sequences(text_to_seq, maxlen = max_sent_len, padding='pre')
    return text_pad_sequences

#For English text - Tokenizer
eng_tokenizer = create_tokenizer(eng)
eng_vocab_size = len(eng_tokenizer.word_index)+1
max_eng_sent_len = max_sentence_length(eng)

#For French Text - Tokenizer
fra_tokenizer = create_tokenizer(fra)
fra_vocab_size= len(fra_tokenizer.word_index)+1
max_fra_sent_len = max_sentence_length(fra)


# max_eng_sent_len = 25
# max_fra_sent_len = 25

#Perform encoding of sequences
X = encode_sequences(eng_tokenizer, eng, max_eng_sent_len)
y = encode_sequences(fra_tokenizer, fra, max_fra_sent_len)

#Train test split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

es = EarlyStopping(monitor='val_accuracy',patience=5,mode='max',verbose=1)
lr = ReduceLROnPlateau(monitor='val_accuracy',patience=3,mode='max',verbose=1,factor=0.1,min_lr=0.001)

def create_model3(inp_vocab_size, out_vocab_size, inp_maxlen, out_maxlen):
    model = Sequential()
    model.add(Embedding(inp_vocab_size, 512,input_length = inp_maxlen, mask_zero=True))
    model.add(Bidirectional(GRU(512)))
    model.add(RepeatVector(out_maxlen))
    model.add(Bidirectional(GRU(512,return_sequences=True)))
    model.add(TimeDistributed(Dense(1024,activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(out_vocab_size,activation='softmax')))
    return model
model3 = create_model3(eng_vocab_size, fra_vocab_size, max_eng_sent_len, max_fra_sent_len)
model3.compile(loss=SparseCategoricalCrossentropy(),optimizer='adamax',metrics='accuracy')

history3 = model3.fit(X_train,
                    y_train.reshape(y_train.shape[0],y_train.shape[1],1),
                    epochs=10,
                    batch_size=512,
                    callbacks=[es,lr],
                    validation_data = (X_test,y_test.reshape(y_test.shape[0],y_test.shape[1],1))
                   )


plt.figure(figsize=(12,8))
plt.plot(history3.history['loss'],'r',label='train loss')
plt.plot(history3.history['val_loss'],'b',label='test loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend();
plt.show()

plt.figure(figsize=(12,8))
plt.plot(history3.history['accuracy'],'r',label='train accuracy')
plt.plot(history3.history['val_accuracy'],'b',label='test accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend();
plt.show()

score=model3.evaluate(X_test,y_test.reshape(y_test.shape[0],y_test.shape[1],1))
print('Test loss:', score[0])
print('Test accuracy:', score[1])


def __input(x, y, x_tk, y_tk, model, y_id_to_word):
    try:
        sentence = input("Type a sentence in English:\n")
        sentence= encode_sequences(x_tk, sentence.split(), max_eng_sent_len)
        sentences = np.array([sentence[0], x[0]])
        print("sentences ", sentences)
        predictions = model.predict(sentences)
        print('translating: ')
        print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    except:
        print("Error! can't translate")
    finally:
        print("End of translation")


y_id_to_word = {value: key for key, value in fra_tokenizer.word_index.items()}
x_id_to_word = {value: key for key, value in eng_tokenizer.word_index.items()}
y_id_to_word[0] = ''
print("y_id_to_word",y_id_to_word)
print("x_id_to_word",x_id_to_word)
while (True):
    __input(X, 0, eng_tokenizer, 0, model3, y_id_to_word)