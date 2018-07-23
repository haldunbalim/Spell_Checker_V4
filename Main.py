import os

NUM_ITERATIONS=100


if __name__=='__main__':
    for i in range(NUM_ITERATIONS):
        os.system('python3 Checker.py')
        os.remove("models/keras_spell_init.h5")
        os.rename("models/keras_spell_e29.h5","models/keras_spell_init.h5")