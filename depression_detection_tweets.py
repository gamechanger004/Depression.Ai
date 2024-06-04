import nltk
nltk.download('punkt')
from TweetModel import TweetClassifier, process_message
from math import log, sqrt
import pandas as pd
import numpy as np
import pickle

class DepressionDetection:

    # Initializing the class and loading the dataset
    def __init__(self):
        self.tweets = pd.read_csv('dataset/tweets.csv')
        self.tweets.drop(['Unnamed: 0'], axis = 1, inplace = True)  # Removing unnecessary column
        self.tweets['label'].value_counts()  # Displaying label counts (for depressive and positive tweets)
        self.tweets.info()  # Displaying dataset info

        self.totalTweets = 8000 + 2314  # Total number of tweets

        # Splitting data into training and testing sets
        trainIndex, testIndex = list(), list()
        for i in range(self.tweets.shape[0]):
            if np.random.uniform(0, 1) < 0.98:
                trainIndex += [i]  # 98% of data for training
            else:
                testIndex += [i]  # 2% of data for testing

        self.trainData = self.tweets.iloc[trainIndex]  # Training data
        self.testData = self.tweets.iloc[testIndex]  # Testing data
        self.trainData['label'].value_counts()  # Displaying label counts in training data
        self.testData['label'].value_counts()  # Displaying label counts in testing data

    # Function for classifying a message as depressive or positive using either 'tf-idf' or 'bow' method
    def classify(processed_message, method):

        # Loading pre-trained data from pickled files
        pickle_in = open("data1.pickle", "rb")
        prob_depressive = pickle.load(pickle_in)
        sum_tf_idf_depressive = pickle.load(pickle_in)
        prob_positive = pickle.load(pickle_in)
        sum_tf_idf_positive = pickle.load(pickle_in)
        prob_depressive_tweet = pickle.load(pickle_in)
        prob_positive_tweet = pickle.load(pickle_in)

        pickle_in = open("data2.pickle", "rb")
        depressive_words = pickle.load(pickle_in)
        positive_words = pickle.load(pickle_in)

        pDepressive, pPositive = 0, 0.  # Initializing probabilities for depressive and positive tweets

        for word in processed_message:
            # Calculating probability for depressive tweets
            if word in prob_depressive:
                pDepressive += log(prob_depressive[word])
            else:
                if method == 'tf-idf':
                    pDepressive -= log(sum_tf_idf_depressive + len(list(prob_depressive.keys())))
                else:
                    pDepressive -= log(depressive_words + len(list(prob_depressive.keys())))

            # Calculating probability for positive tweets
            if word in prob_positive:
                pPositive += log(prob_positive[word])
            else:
                if method == 'tf-idf':
                    pPositive -= log(sum_tf_idf_positive + len(list(prob_positive.keys())))
                else:
                    pPositive -= log(positive_words + len(list(prob_positive.keys())))

            pDepressive += log(prob_depressive_tweet)  # Adding overall depressive tweet probability
            pPositive += log(prob_positive_tweet)  # Adding overall positive tweet probability

        # Returning the label with the higher probability
        if pDepressive >= pPositive:
            return 1  # Depressive
        else:
            return 0  # Positive

    # Function to calculate and print evaluation metrics
    def metrics(self, labels, predictions):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        for i in range(len(labels)):
            true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
            true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
            false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
            false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)

# Main function to execute the classification and evaluation
if __name__ == "__main__":
    obj = DepressionDetection()  # Creating an instance of DepressionDetection
    sc_tf_idf = TweetClassifier(obj.trainData, 'tf-idf')  # Initializing TweetClassifier for TF-IDF
    #sc_tf_idf.train()  # Training the classifier (commented out)
    preds_tf_idf = sc_tf_idf.predict(obj.testData['message'], 'tf-idf')  # Predicting using TF-IDF
    obj.metrics(obj.testData['label'], preds_tf_idf)  # Evaluating the predictions

    sc_bow = TweetClassifier(obj.trainData, 'bow')  # Initializing TweetClassifier for BOW
    #sc_bow.train()  # Training the classifier (commented out)
    preds_bow = sc_bow.predict(obj.testData['message'], 'bow')  # Predicting using BOW
    obj.metrics(obj.testData['label'], preds_bow)  # Evaluating the predictions

    """# Predictions with TF-IDF
    # Depressive Tweets
    """
    pm = process_message('Extreme sadness, lack of energy, hopelessness')  # Processing a depressive message
    print(f"Extreme sadness, lack of energy, hopelessness : {sc_tf_idf.classify(pm, 'tf-idf')}")  # Classifying and printing the result

    """# Positive Tweets"""
    pm = process_message('Loving how me and my lovely partner is talking about what we want.')  # Processing a positive message
    print(f"Loving how me and my lovely partner is talking about what we want. : {sc_tf_idf.classify(pm, 'tf-idf')}")  # Classifying and printing the result

    """# Predictions with Bag-of-Words (BOW)
    # Depressive tweets """
    pm = process_message('Hi hello depression and anxiety are the worst')  # Processing a depressive message
    sc_bow.classify(pm, 'bow')  # Classifying using BOW

    """# Positive Tweets"""
    pm = process_message('Loving how me and my lovely partner is talking about what we want.')  # Processing a positive message
    sc_bow.classify(pm, 'bow')  # Classifying using BOW
