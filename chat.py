import json
import pickle
import random
import webbrowser
import numpy as np
from datetime import datetime
from keras.models import load_model
from nlp_utility import lemmatize, tokenize_n_filter, bag_of_words


model = load_model("model/ANN_engine.h5")
intents = json.load(open("data/intents.json", "rb"))
all_words = pickle.load(open("model/all_words.pkl", "rb"))
tag_classes = pickle.load(open("model/tag_classes.pkl", "rb"))


def datetime_process(*usr_inp):
  return datetime.strftime(datetime.now(), "%Y-%m-%d %a || %I:%M %p")

def news_process(*usr_inp):
  return "Current News: "

def google_search(*usr_inp):
  # print(usr_inp)
  if len(usr_inp) > 0:
    usr_inp = " ".join(usr_inp[0]).replace("google", "").replace("internet", "").replace("search", "")

    webbrowser.open(f"https://www.google.com/search?q={usr_inp}")

    return f"Searching for: {usr_inp}"


num_cols = int(open("model/num_cols.txt", "r").read().strip())
# print(num_cols)

def chatting():
  run = True
  custom_actions = {'datetime': datetime_process, "news": news_process, "google": google_search}


  while run:
    usr_inp = tokenize_n_filter(lemmatize(input("\nYou    : ")))

    bag = bag_of_words(usr_inp, all_words).reshape((1, num_cols))
    probs = model.predict(bag, verbose=0)

    if np.any(probs >= 0.70):
      tag_idx = list(map(np.argmax, probs))[0]
      tag = tag_classes[tag_idx]

      for intent in intents["intents"]:
        if tag in custom_actions.keys():
          print("Cybrix: ", custom_actions[tag](usr_inp), end="\n")
          break
        elif intent["tag"] == tag:
          print("Cybrix :", random.choice(intent["responses"]), end="\n")

        if tag == "goodbye":
          run = False

    else:
        print("Cybrix :", random.choice(intents["noanswer"]), end="\n")
  print()


if __name__ == "__main__":
  chatting()


