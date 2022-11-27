import pickle
import re
import collections
import os

STOPWORDS = ['amazon', 'amazoncom', 'amazonca', 'amazoncouk', 'amazonde', 'amazonfr', 'amazonit', 'amazones',
             'amazonca']


class Predictor:
    def __init__(self, training_data):
        self.train = None  # doesn't actually read in training data until we need to

        self.unigrams = []
        self.bigrams = []
        self.trigrams = []
        self.fourgrams = []
        self.fivegrams = []

        files_to_load = [
            'saved/unigrams.pkl', 'saved/bigrams.pkl', 'saved/trigrams.pkl', 'saved/fourgrams.pkl',
            'saved/fivegrams.pkl'
        ]

        self.initialize_dicts(files_to_load, training_data)

    # initialize_dicts() loads in the data from the ngram files in the "saved" folder if they exists; Otherwise, it invokes generate_ngrams() on the training data from the "train.txt" file in the "data" folder
    def initialize_dicts(self, files_to_load, training_data):
        count = 0

        for path in files_to_load:
            if os.path.exists(path):
                print("Loading dicts for " + str(count + 1) + "-grams from file...")
                with open(path, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    if count == 0:
                        self.unigrams = unpickler.load()
                    elif count == 1:
                        self.bigrams = unpickler.load()
                    elif count == 2:
                        self.trigrams = unpickler.load()
                    elif count == 3:
                        self.fourgrams = unpickler.load()
                    elif count == 4:
                        self.fivegrams = unpickler.load()
            else:
                print("Generating dicts on " + str(count + 1) + "-grams, this may take a few minutes...")
                if self.train is None:
                    print("Reading in training data...")
                    self.train = self.read_data(training_data)
                if count == 0:
                    self.unigrams = self.generate_ngrams(self.train, 1)
                    with open(path, 'wb') as f:
                        pickle.dump(self.unigrams, f, pickle.HIGHEST_PROTOCOL)
                elif count == 1:
                    self.bigrams = self.generate_ngrams(self.train, 2)
                    with open(path, 'wb') as f:
                        pickle.dump(self.bigrams, f, pickle.HIGHEST_PROTOCOL)
                elif count == 2:
                    self.trigrams = self.generate_ngrams(self.train, 3)
                    with open(path, 'wb') as f:
                        pickle.dump(self.trigrams, f, pickle.HIGHEST_PROTOCOL)
                elif count == 3:
                    self.fourgrams = self.generate_ngrams(self.train, 4)
                    with open(path, 'wb') as f:
                        pickle.dump(self.fourgrams, f, pickle.HIGHEST_PROTOCOL)
                elif count == 4:
                    self.fivegrams = self.generate_ngrams(self.train, 5)
                    with open(path, 'wb') as f:
                        pickle.dump(self.fivegrams, f, pickle.HIGHEST_PROTOCOL)
            count = count + 1

    # generate_ngrams() takes in text data and a desired n_gram size and uses a sliding window to create a dictionary of data {count:n_gram}
    def generate_ngrams(self, data, ngram_size):
        result = {}
        for line in data:  # For each line of the training data
            words = line.split(' ')  # Split each line into words
            words_cleaned = []
            for word in words:
                if word not in STOPWORDS:
                    words_cleaned.append(word)
            if len(words_cleaned) >= ngram_size:  # if there are enough words to generate an n-gram of this size
                for i in range(len(words_cleaned) - ngram_size + 1):  # For each n-gram of this size in this line
                    ngram = words_cleaned[i]  # initializes the n-gram to the first word in this range
                    for n in range(ngram_size - 1):
                        ngram = ngram + ' ' + words_cleaned[n + i + 1]  # concatenate the rest of the words
                    result[ngram] = result.get(ngram, 0) + 1  # put in map and update occurrences of this n-gram
        return result

    #  Reads in a txt file, line by line into a list
    #  Each index of the list is one line of the .txt file
    def read_data(self, data_file):
        with open(data_file, "r", encoding="utf8") as f:  # opens the data_file
            # This state reads in each line of the data file
            # All characters are made lowercase
            # We use regex to remove any characters that are not a whitespace character, or a letter or digit
            result = []
            for lines in f:
                lines = lines.splitlines()
                for line in lines:
                    result.append(re.sub(r'[^\w\s]', '', line.strip().lower()))
            return result

    #  predict_next_word() returns a predicted next word given an input string using the generated n_gram data with backoff and smoothing
    def predict_next_word(self, user_input, print_message=False):
        split_input = user_input.lower().split(' ')  # make input lowercase and split it by spaces
        highest_occurrences = -1
        best_word = ""

        if len(split_input) >= 4:
            split_input = split_input[len(split_input) - 4: len(split_input)]
            ngrams = self.fivegrams
        if len(split_input) >= 3:
            ngrams = self.fourgrams
        elif len(split_input) == 2:
            ngrams = self.trigrams
        elif len(split_input) == 1:
            ngrams = self.bigrams
        else:
            print("Error, must have at least 1 word to predict next word")
            return

        for word in self.unigrams:  # for each word we've seen in training data
            key = ' '.join(split_input) + ' ' + word  # add this word to the end of our input
            occurrences = ngrams.get(key, None)

            if occurrences is not None and occurrences > highest_occurrences:  # if this bigram occurs more that our current max
                highest_occurrences = occurrences  # new best occurrences
                best_word = word  # this is our new most likely word

        # if unidentified word in input, attempt with smaller n-gram
        if best_word == "":
            return self.predict_next_word(' '.join(split_input[1:]), print_message)

        #  If the optional print_message boolean is passed in as True, print the input/output
        if print_message:
            print("Input = '" + str(user_input) + "'")
            print("Predicted next word = '" + best_word + "'")
        return best_word

    # n_most_common_n_grams is a visualization function used during development. It returns the given n most common n grams in the data. This function does not affect word prediction of our model.
    def n_most_common_n_grams(self, n, n_gram_size):
        if n_gram_size > 5 or n_gram_size < 1:
            print("n gram size too large")

        if n_gram_size == 1:
            d = collections.Counter(self.unigrams)
        elif n_gram_size == 2:
            d = collections.Counter(self.bigrams)
        elif n_gram_size == 3:
            d = collections.Counter(self.trigrams)
        elif n_gram_size == 4:
            d = collections.Counter(self.fourgrams)
        elif n_gram_size == 5:
            d = collections.Counter(self.fivegrams)

        print()
        print("Top " + str(n) + " most common " + str(n_gram_size) + "-grams:")
        for thing in d.most_common(n):
            print(str(thing[1]) + " " + thing[0])

    #  Generates a sentence given at least one word (can be more), and the desired number of words to add
    def generate_sentence(self, sentence_start, length):
        sentence = sentence_start.lower()
        for i in range(length):
            sentence = sentence + ' ' + self.predict_next_word(sentence)

        print(sentence)

    # test will evaluate the accuracy of the mord prediction model on the given test_data and return the accuracy in terms of %
    def test(self, test_data):
        trainingData = self.read_data(test_data)
        count = 0
        corrCount = 0
        # evaluate on first 100 predictions for demonstration sake. Can be increased by removing below line
        trainingData = trainingData[0:100]
        for line in trainingData:
            splitLine = line.split(" ")
            prediction = self.predict_next_word(" ".join(splitLine[0:-1]), False)
            actual = splitLine[-1]
            print("Prediction: " + prediction + " Actual: " + actual)
            count += 1
            if (prediction == actual):
                corrCount += 1
        return (corrCount / count) * 100

    # demo_word enters a loop which prompts the user for an input string and then uses the model to predict the next word.
    def demo_word(self):
        while True:
            user_input = str(input('Enter a word or a phrase (Enter "%" to stop):'))
            if user_input.strip() == '%':
                break
            user_input = re.sub(r'[^\w\s]', '', user_input)  # remove characters that are not digits, letters, or whitespace
            print('Predicted next word: ' + self.predict_next_word(user_input.lower().strip()))

    # demo_sentence enters a loop which prompts the user for an input string and then uses the model to predict the next 40 words.
    def demo_sentence(self):
        while True:
            user_input = str(input('Enter a word or a phrase (Enter "%" to stop):'))
            if user_input.strip() == '%':
                break
            user_input = re.sub(r'[^\w\s]', '', user_input)  # remove characters that are not digits, letters, or whitespace
            self.generate_sentence(self.predict_next_word(user_input.lower().strip()), 40)
