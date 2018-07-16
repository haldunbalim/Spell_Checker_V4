from speller import spell
from random import shuffle
import numpy as np


from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, recurrent, Embedding
from keras.callbacks import Callback
from keras import  optimizers

FILE_FOLDER_NAME = "text files/"
SPELL_INDEX_FILE = FILE_FOLDER_NAME+"spell_index.txt"
WORDS_FILE = FILE_FOLDER_NAME+"full_short.txt"
MAXIMUM_LEN = 10
LEARNING_RATE=0.001

RANDOM_RATE = 0.7
RANDOM_TIMES = 2

AMOUNT_OF_DROPOUT = 0.2
INITIALIZATION = "he_normal"
HIDDEN_SIZE = 256
NUM_ITERATIONS = 1

BATCH_SIZE = 128
NUM_EPOCHS = 1000
STEPS_PER_EPOCH = 100
SAVED_MODEL_FILE_NAME="models/keras_spell_e{}.h5"
LOAD_MODEL_FILE_NAME="models/keras_spell_e7.h5"
NUM_SAMPLES_ON_CALLBACK=5
NUM_INPUT_LAYERS=1
NUM_OUTPUT_LAYERS=1

def shuffle_word(word):
    r= np.random.randint(len(word)-1)
    d=word[0:r]
    d+=word[r+1]
    d+=word[r]
    d+=word[r+2:len(word)]
    return d

def remove_char(word):
    r= np.random.randint(1,len(word))
    d=word[0:r]
    d+=word[r+1:len(word)]
    return d

def add_char(word):
    r= np.random.randint(1,len(word))
    random_char=chr(np.random.randint(ord('a'),ord('z')+1))
    d=word[0:r]
    d+=random_char
    d+=word[r:len(word)]
    return d


def typo_generator(s,random_rate=0.6,times=0):
    d= ""
    for word in s.split():
        r= np.random.rand()
        if(r>1-random_rate and len(word)>2):
            r=np.random.randint(0,3)
            if(r==1):
                d+=shuffle_word(word)+' '
            elif(r==2):
                d+=add_char(word)+' '
            else:
                d+=remove_char(word)+' '
        else:
            d+=word+' '
    if times==0:
        return d[:-1]
    else:
        return typo_generator(d[:-1],random_rate,times-1)


class spell_index:

    def __init__(self):
        self.ind_to_spell = {}
        self.spell_to_ind = {}
        self.fill_maps()
        self.num_spells = len(self.ind_to_spell)

    def fill_maps(self):
        with open(SPELL_INDEX_FILE, 'r') as file:
            lines = file.readlines()
            shuffle(lines)
            for line in lines:
                temp = line.split()
                spell = temp[0]
                index = int(temp[1])
                self.ind_to_spell[index] = spell
                self.spell_to_ind[spell] = index

    def get_index(self, spell):
        return self.spell_to_ind[spell]

    def get_spell(self, index):
        return self.ind_to_spell[index]



def _vectorize(word,padding=False):
    current=[]
    if padding:
        current.append(indexer.get_index("GO"))
        for s in spell(word):
            try:
                current.append(indexer.get_index(s))
            except:
                current.append(indexer.get_index("UNKNOWN"))
        current.append(indexer.get_index("END"))
        return current
    else:
        for s in spell(word):
            try:
                current.append(indexer.get_index(s))
            except:
                current.append(indexer.get_index("UNKNOWN"))
        return current

def vectorize(word):
    vec=_vectorize(word)
    result=np.zeros([MAXIMUM_LEN,indexer.num_spells])
    if len(vec) <= MAXIMUM_LEN:
        for ind,el in enumerate(vec):
            result[ind][el]=1
        return result
    else:
        return result


def devectorize(vector, one_hot=False,padding=False):
    if one_hot:
        1+1
    if padding:
        1+1

    st=""

    for i in range(MAXIMUM_LEN):
        current = np.argmax(vector[i, :])

        if current==0:
            return st

        try:
            current=indexer.get_spell(current)
        except:
            current="**"

        if current=="UNKNOWN":
            st+="**"
        else:
            st += current
    return st


def print_random_predictions(model):
    """Select 10 samples from the validation set at random so we can visualize errors"""
    print()
    for _ in range(NUM_SAMPLES_ON_CALLBACK):
        Q,A,V= ds.random_sample()
        V=np.array([V[0]])
        preds = model.predict(V)

        guess=devectorize(preds[0])

        print('Q:', Q)
        print('A:', A)
        print("P:", guess)
        print('---')
    print()

class OnEpochEndCallback(Callback):
    """Execute this every end of epoch"""

    def on_epoch_end(self, epoch, logs=None):
        """On Epoch end - do some stats"""
        print_random_predictions(self.model)
        self.model.save(SAVED_MODEL_FILE_NAME.format(epoch))




class dataset:

    def __init__(self):
        self.data = []
        self.fill_data()
        print("Data is read")

        self.data_vectors = []
        self.fill_data_vectors()
        print("Data is vectorized")

        self.num_samples = len(self.data)
        self.current = 0
        print("Dataset is created")

    def fill_data(self):
        with open(WORDS_FILE) as file:
            for line in file.readlines():
                word = line.split()[0]
                self.data.append((typo_generator(word, random_rate=RANDOM_RATE, times=RANDOM_TIMES), word))
        shuffle(self.data)

    def fill_data_vectors(self):
        long_ls=[]

        for ind, el in enumerate(self.data):
            vec_x = vectorize(el[0])
            vec_y = vectorize(el[1])
            if vec_x.any() != 0 or vec_y.any() != 0:
                self.data_vectors.append((vec_x, vec_y))
            else:
                long_ls.append(ind)

    def next_batch(self, batch_size):
        if self.current + batch_size >= self.num_samples:
            self.reset()
        self.current += batch_size
        ls = self.data_vectors[self.current - batch_size:self.current]
        return np.array([el[0] for el in ls]), np.array([el[1] for el in ls])

    def reset(self):
        self.current = 0
        self.fill_data()
        self.fill_data_vectors()

    def generator(self):
        while True:
            yield self.next_batch(BATCH_SIZE)

    def random_sample(self):
        ind = np.random.randint(0, len(self.data))
        a=self.data[ind][0]
        b=self.data[ind][1]
        c=self.data_vectors[ind]
        return a, b, c




def iterate_training(model, X_train, y_train):
    """Iterative Training"""
    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(0, NUM_ITERATIONS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)


def generate_model():
    model = Sequential()



    for i in range(NUM_INPUT_LAYERS):
        model.add(recurrent.GRU(HIDDEN_SIZE, input_shape=(None, indexer.num_spells),
                                kernel_initializer = INITIALIZATION, return_sequences=i+1<NUM_INPUT_LAYERS))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    model.add(RepeatVector(MAXIMUM_LEN))

    for i in range(NUM_OUTPUT_LAYERS):
        model.add(recurrent.GRU(HIDDEN_SIZE,return_sequences=True,kernel_initializer=INITIALIZATION))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    model.add(TimeDistributed(Dense(indexer.num_spells, kernel_initializer=INITIALIZATION)))
    model.add(Activation('softmax'))

    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



def iterative_train(model):
    model.fit_generator(ds.generator(), steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        callbacks=[ON_EPOCH_END_CALLBACK, ],
                        class_weight=None, max_q_size=10, workers=1,
                        pickle_safe=False, initial_epoch=0)


def train_speller(from_file=None):
    """Train the speller"""
    if from_file:
        model = load_model(from_file)
    else:
        model = generate_model()
    iterative_train(model)


if __name__ == '__main__':
    ON_EPOCH_END_CALLBACK = OnEpochEndCallback()
    indexer = spell_index()
    ds = dataset()
    train_speller(LOAD_MODEL_FILE_NAME)