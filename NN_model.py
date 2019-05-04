import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Dense, LSTM, Bidirectional, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import tensorflowjs as tfjs
import tensorflow as tf

def separate_seq(seqs, n=3):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs])

df = pd.read_csv("data.csv")

aa_seqs = df['seq']
dssp_seqs = df['sst3']

#measurer = np.vectorize(len)
#print(measurer(dssp_seqs.values.astype(str)).max(axis=0))
#max length of the sequences are 5037

max_length = 256 #limit the max length

aa_seqs = separate_seq(aa_seqs)

aa_tokenizer = Tokenizer()
aa_tokenizer.fit_on_texts(aa_seqs)
aa_data = aa_tokenizer.texts_to_sequences(aa_seqs)
aa_data = pad_sequences(aa_data, maxlen=max_length, padding='post', truncating='post')

print(aa_tokenizer.word_index)

dssp_tokenizer = Tokenizer(char_level = True)
dssp_tokenizer.fit_on_texts(dssp_seqs)
dssp_data = dssp_tokenizer.texts_to_sequences(dssp_seqs)
dssp_data = pad_sequences(dssp_data, maxlen=max_length, padding='post', truncating='post')
dssp_data = to_categorical(dssp_data)

print(dssp_tokenizer.word_index)

aa_train, aa_test, dssp_train, dssp_test = train_test_split(aa_data, dssp_data, test_size=0.2, random_state=0)
aa_train, aa_val, dssp_train, dssp_val = train_test_split(aa_train, dssp_train, test_size=0.2, random_state=0)

def mask_acc(y_true, y_pred):
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    ignore_mask = K.cast(K.not_equal(y_true_class, 0), "int32")
    matches = K.cast(K.equal(y_true_class, y_pred_class), "int32") * ignore_mask
    accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
    return accuracy

def create_model():
    model = Sequential()
    model.add(Embedding(len(aa_tokenizer.word_index)+1, 256, input_length=max_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(len(dssp_tokenizer.word_index)+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[mask_acc])
    return model

model = create_model()
model.summary()
model.fit(aa_train, dssp_train, batch_size=max_length, validation_data=(aa_val, dssp_val), epochs = 5)

aa_dict = {value:key for key,value in aa_tokenizer.word_index.items()}
dssp_dict = {value:key for key,value in dssp_tokenizer.word_index.items()}

def dssp_to_seqs(output):
    dssp = ""
    for prob in output:
        i = np.argmax(prob)
        if i != 0:
            dssp += dssp_dict[i]
        else:
            break
    return dssp

def aa_to_seqs(input):
    aa = ""
    for i in input:
        if i != 0:
            aa += aa_dict[i]
        else:
            break
    return aa


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
tfjs.converters.save_keras_model(model, "new_model.json")
print("Saved model to disk")

dssp_pred = model.predict(aa_test)
print('Testing')
for i in range(3):
    print("AA sequence:")
    print(aa_to_seqs(aa_test[i]).upper())
    print("Predicted DSSP sequence:")
    print(dssp_to_seqs(dssp_pred[i]).upper())
    print("True DSSP sequence:")
    print(dssp_to_seqs(dssp_test[i]).upper())
    print()

ave = 0
for i in range(len(dssp_pred)):
    ave += mask_acc(dssp_test[i], dssp_pred[i])
print("Testing set accuracy: ")
tf.print(ave)

'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''
