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


class InputText(object):
    """
    The InputText class cleans up the input before sending into the model.
    """

    def __init__(self, texts):
        """
        Defines the texts.
        """
        self.texts = texts

    def text_cleaning(self):
        """
        Splits the input text and stores them in a list

        :rtype: List[str]
        """
        texts = self.texts
        texts.strip("'").strip('"')

        lines = texts.split('\r')
        line_list = []
        for line in lines:
            if line.strip("\n").strip(" ") != "":
                line = line.replace("\n", ' ')
                line = line.replace("\t", ' ')
                line = line.replace("\\", ' ')
                line = line.replace("\"", ' ')
                line = line.replace("*", ' ')
                line_list.append(line)
        return line_list


class Predictor(object):
    """
    The Predictor using Classifier to predict the tag of the input text.
    """

    def __init__(self, sentence):
        """
        Defines the sentence.
        """
        self.sentence = sentence

    def predict(self):
        """
        Implements the predictions
        The first float is the probability of being tagged as 0.
        The Second float is the probability of being tagged as 1.
        The str is the orginal text.
        :rtype: Dictionary{str: List[List[float, float, str]]}
        """
        _input = self.sentence
        contract = InputText(_input)
        sentence = contract.text_cleaning()

        if sentence:
            untagged_sentence = Classifier(sentence)
            untagged_sentence.sentence_encode()
            prob = untagged_sentence.sentence_predict()

        result = []
        for i in range(len(sentence)):
            result.append([prob[i][0], prob[i][1], sentence[i]])

        body = {"result": result}
        return body
