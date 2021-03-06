import sys
import os.path
import numpy
import tensorflow as tf
from networkSize import get_model_memory_usage
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.utils import np_utils
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')

def tokenize(input):
    input = input.lower() # To reduce possible characters
    tokenizer = RegexpTokenizer(r'\w+')
    print("Tokenizing...")
    tokens = tokenizer.tokenize(input)
    print("Input tokenized ", len(tokens), " found")
    # Remove all tokens containing stopwords
    filtered = filter(lambda tok: tok not in stopwords.words('english'), tokens)
    print("Tokens filtered")
    return " ".join(filtered)
    
def prepare(data):
    seq_len = 100
    x_data = []
    y_data = []
    
    
    for i in range(0, input_len - seq_len, 1):
        #Convert every char into numbers
        in_seq = processed[i:i + seq_len]
        out_seq = processed[i + seq_len]
    
        x_data.append([char_num[char] for char in in_seq])
        y_data.append(char_num[out_seq])
        if i % 1000000 == 0:
            print(i)
        
       
    print ("Total Patterns:", len(x_data))
    
    X = numpy.reshape(x_data, (len(x_data), seq_len, 1))
    X = X/float(vocab_len) #Convert into floats
    
    y = np_utils.to_categorical(y_data) # Convert the label data
    return (X,y) 

def train(data, model, epoch):
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_call = [checkpoint] #Saves time for second execution
    '''
    tensorboard_callback = TensorBoard(
    log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
    write_grads=False, write_images=False, embeddings_freq=0,
    embeddings_layer_names=None, embeddings_metadata=None,
    embeddings_data=None, update_freq='epoch')
    '''
    model.fit(data[0], data[1], epochs=epoch, batch_size=64, callbacks=[desired_call]) #Train the model


processed = ""
for i in range (1,1):   
    file = open("wp_" + str(i) + ".txt", encoding="utf-8").read()
    processed = processed + file
#processed = tokenize(file)
processed = open("wp_1.txt", encoding="utf-8").read()
chars = sorted(list(set(processed)))
char_num = dict((c, i) for i, c in enumerate(chars)) # Convert chars in numbers

input_len = len(processed)
vocab_len = len(chars)
print ("Total number of chars:", input_len)
print ("Total vocab:", vocab_len)




filepath = "model_saved.hdf5"

if os.path.isfile(filepath):
    model=load_model(filepath)
else:
    #Model creation
    model = Sequential()
    model.add(LSTM(256, batch_input_shape=(64, 100, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(42, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()
print(get_model_memory_usage(256, model))
for i in range(10):
    for j in range(1,20):
        file = open("wp_" + str(j) + ".txt", encoding="utf-8").read()
        (X,y) = prepare(file)
        train((X,y), model, 1)


#Testing model
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
    index = numpy.random.choice(len(prediction), p=prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
