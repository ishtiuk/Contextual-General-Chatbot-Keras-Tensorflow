# Contextual-General-Chatbot-Keras-Tensorflow
# CYBRIX Contextual General Chatbot

CYBRIX is a general chatbot that can converse on various topics. It is not highly sophisticated but it will try its best to understand and respond to your prompts.

## Architecture

The model is based on a simple ANN Feed Forward Neural Network with 5 hidden layers and 128 neurons (perceptrons). It is mainly trained to predict the class or tag of the user's message. For example, a message like "hi" or "hello" will have the class of "greetings". The model is trained on the "intents.json" file, which contains general chat patterns and their corresponding responses.

## Functions

The following functions are used in the chatbot:

- `tokenize_n_filter(word_lst)`: This function tokenizes the text and filters out the punctuations from the text.
- `lemmatize(text)`: This function lemmatizes the text or sentence with the Spacy library's "en_core_web_sm" model.
- `bag_of_words(tokenized_sentence, all_words)`: This function converts the user's message to a bag-of-words encoded vector.

## Usage

To use CYBRIX, simply run the `main.py` file and start chatting with the bot. The bot will predict the class or tag of your message and respond with a random response from the "intents.json" file.

## Limitations

Since the model is based on a simple ANN Feed Forward Neural Network, it may not be able to understand complex prompts or messages. However, it will try its best to respond appropriately.

## Contributions

Contributions to the project are welcome. Please feel free to submit a pull request or open an issue if you have any suggestions or feedback.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
