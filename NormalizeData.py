from Parameters import save_training_data_file , normalized_training_data_file

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

training_data = list(np.load(save_training_data_file,allow_pickle=True))

df = pd.DataFrame(training_data)
print(df.head())
print(Counter(df[1].apply(str)))

# Only shuffling the data
shuffle(training_data)
np.save(normalized_training_data_file, training_data)



'''
# Sentdex way of normalizing his data. Will ignore for now 

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')


forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data_v2.npy', final_data)
'''