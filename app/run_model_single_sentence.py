from model import Classifier
import sys

"""
This python file provides the simple way to see the predict result of a single sentence using the combine model of SVM and BERT.
"""


def single_sentence_test():
    """
    type: List[str]
    """
    lines = sys.stdin.readlines()
    target = lines[0]
    target.strip("'").rstrip("\n")
    sentence = Classifier([target])
    sentence.sentence_encode()
    result = sentence.sentence_predict()
    print(f"tag 0: { result[0][0]*100:.2f} %")
    print(f"tag 1: { result[0][1]*100:.2f} %")

    return


if __name__ == "__main__":
    single_sentence_test()
