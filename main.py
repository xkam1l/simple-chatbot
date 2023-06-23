from json import load as json_load
from random import choice as random_choice

import tensorflow as tf
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from numpy import argmax as np_argmax
from numpy import array as np_array
from tflearn import DNN
from tflearn import fully_connected
from tflearn import input_data
from tflearn import regression

stemmer = LancasterStemmer()

if __name__ == '__main__':
    with open("intents.json") as file:
        data = json_load(file)

    words = list()
    labels = list()
    docs_x = list()
    docs_y = list()

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = list()
    output = list()

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        [bag.append(1 if w in wrds else 0) for w in words]

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np_array(training)
    output = np_array(output)

    tf.compat.v1.reset_default_graph()

    net = input_data(shape=[None, len(training[0])])
    net = fully_connected(net, 8)
    net = fully_connected(net, 8)
    net = fully_connected(net, len(output[0]), activation="softmax")
    net = regression(net)

    model = DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)


    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]

        s_words = word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np_array(bag)


    def chat():
        print("The bot is here to answer your questions (exit/quit to exit)")
        print('-------------------')
        print()

        while True:
            inp = input('You: ')
            if inp.lower() in ['quit', 'exit']:
                break

            results = model.predict([bag_of_words(inp, words)])
            results_index = np_argmax(results)
            tag = labels[results_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(f'Bot: {random_choice(responses)}')


    chat()
