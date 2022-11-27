from predictor import Predictor

if __name__ == '__main__':
    data = 'data/' + 'train.txt'
    predictor = Predictor(data)
    print("Predict next single word given user input.")
    predictor.demo_word()
    print("Generate a sentence given user input.")
    predictor.demo_sentence()

    sentence_starters = ['would you rather', 'the most', 'apples and', 'what', 'pineapple', 'watermelon', 'pineapples']
    print("Testing Accuracy of model on starter sentences from \"test.txt\" (n=100). This process takes a couple of minutes.")
    print("Model accuracy on predicting exact next word: {}%".format(predictor.test("data/test.txt")))
    print("finished")
