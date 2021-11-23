from flask import Flask, request, render_template
from model import Predictor

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('Home.html')


@app.route('/article', methods=['GET'])
def article():
    return render_template('Article.html')


@app.route('/analysis', methods=['GET'])
def analysis(input=None):
    print('input', input)
    return render_template('Analysis.html', article=str(input))


@app.route("/post_submit", methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # TODO: do analysis from bert, and then pass to analysis page
        res = request.form.get('article')
        res = Predictor(res)
        res = res.predict()
        return analysis(res)


if __name__ == '__main__':
    app.debug = True
    app.run()
