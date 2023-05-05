import nltk
import spacy
import numpy as np
from nltk.stem import PorterStemmer


nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

####### Tokenize x Filter out punctuations #######

def tokenize_n_filter(word_lst):
  punc = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

  word_lst = nltk.word_tokenize(" ".join(word_lst).lower())
  word_lst = [word for word in word_lst if word not in punc]
  
  return word_lst

####### Lemmatization #######

def lemmatize(text):
  doc = nlp(text)
  lemma_lst = [token.lemma_ for token in doc]

  return lemma_lst
  


####### Bag of Words generating ########

def bag_of_words(tokenized_sentence, all_words):
  vector = np.zeros(len(all_words))

  for idx, elem in enumerate(all_words):
    if elem in tokenized_sentence:
      vector[idx] = 1

  return vector
