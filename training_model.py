import json
import tensorflow
import numpy as np
from nlp_utility import lemmatize, tokenize_n_filter, bag_of_words


intents = json.load(open('data/intents.json', 'rb'))

all_words = []
tag_classes = []          ## y_train
xy = []

for intent in intents['intents']:
  tag = intent['tag']
  tag_classes.append(tag)

  for pattern in intent['patterns']:
    words = tokenize_n_filter(lemmatize(pattern))
    all_words.extend(words)
    xy.append((words, tag))
    

all_words = sorted(list(set(all_words)))
tag_classes = sorted(list(set(tag_classes)))

x_train = []
y_train = []


for (pattern, tag) in xy:
  bag = bag_of_words(pattern, all_words)
  
  x_train.append(bag)
  label = tag_classes.index(tag)                  ## LabelEncoding... ;)
  y_train.append(label)

num_classes = len(set(y_train))
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train)




from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="accuracy",
    patience=15,
    mode="auto",
    restore_best_weights=True
)


model = Sequential()

model.add(Dense(128, activation="relu", input_dim=len(x_train[0])))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation="softmax"))

# model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=50, callbacks=early_stop)


y_pred = model.predict(x_train, verbose=0)
y_pred = np.array(list(map(np.argmax, y_pred)))

print("\n\nAccuracy:", (sum(y_train == y_pred) / len(y_train)) * 100, "%")


### saving Model ###
model_path = "model/ANN_engine.h5"
model.save(model_path)

num_cols = x_train.shape[1]
num_cols_path = "model/num_cols.txt"

print("Saved Model at: [", model_path, "]", sep="")
print("X_train shape: ", x_train.shape)

with open(num_cols_path, "w") as f:
  f.write(str(num_cols))

print("X_train (num_cols) saved at: [", num_cols_path, "]", sep="")


