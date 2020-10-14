import sys
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def tokenize(input):
    input = input.lower() # To reduce possible characters
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # Remove all tokens containing stopwords
    filtered = filter(lambda tok: tok not in stopwords.words('english'), tokens)
    return " ".join(filtered)

file = open("writingPrompts.txt").read()
processed = tokenize(file)

chars = sorted(list(set(processed_inputs)))
char_num = dict((c, i) for i, c in enumerate(chars)) # Convert chars in numbers

input_len = len(processed)
vocab_len = len(chars)
print ("Total number of chars:", input_len)
print ("Total vocab:", vocab_len)


seq_len = 100
x_data = []
y_data = []


for i in range(0, input_len - seq_len, 1):
    #Convert every char into numbers
    in_seq = processed[i:i + seq_len]
    out_seq = processed[i + seq_len]

    x_data.append([char_num[char] for char in in_seq])
    y_data.append(char_num[out_seq])
    
   
print ("Total Patterns:", len(x_data))

X = numpy.reshape(x_data, (n_patterns, seq_len, 1))
X = X/float(vocab_len) #Convert into floats

y = np_utils.to_categorical(y_data) # Convert the label data

#Model cration
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_call = [checkpoint] #Saves time for second execution
model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_call) #Train the model

model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))

start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")


for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
