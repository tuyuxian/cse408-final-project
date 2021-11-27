from flask import Flask, request, render_template
from model import Predictor
from unit_test import UnitTest
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('Home.html')


@app.route('/article', methods=['GET'])
def article():
    return render_template('Article.html')


@app.route('/analysis', methods=['GET'])
def analysis(text):
    #print('input', text)
    return render_template('Analysis.html', article=text)


@app.route("/post_submit", methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        res = request.form.get('text-to-detect')
        # res = Predictor(res)
        # res = res.predict()
        # return analysis(res)
        res = UnitTest()
        return analysis(res.sentence_output)


if __name__ == '__main__':
    app.debug = True
    app.run()
