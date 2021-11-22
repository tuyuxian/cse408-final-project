from bert_serving.client import BertClient
from joblib import load


class Classifier(object):
    """
    The Classifier class load two models.
    1. fine-tune BERT model from bert_serving_client
    2. SVM model from local
    """

    def __init__(self, sentence):
        """
        Defines sentence and loads model.
        """
        self.bc = BertClient()
        self.model = load('./leakjudge.joblib')
        self.sentence = sentence

    def sentence_encode(self):
        """
        Encode the sentence.
        """
        self.sentence_encoded = self.bc.encode(self.sentence)

    def sentence_predict(self):
        """
        Predicts the tag of the input sentence with SVM model.

        The first column of the result_prob represents the probability of the sentence being tagged 0,
        and the second column stands for the probabililty of being tagged as 1.

        :rtype: List[List[float]
        """
        result = self.model.predict(self.sentence_encoded)
        result_prob = self.model.predict_proba(self.sentence_encoded)
        return result_prob
