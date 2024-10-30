from flask import Flask
from flask import url_for, render_template, request, redirect, send_file
import csv
import pandas as pd
from dataset.datalib import SentenceFinder
app = Flask(__name__)
query = SentenceFinder()


@app.route('/')
def index():
    return render_template('index.html')  # главная страница


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        srch = request.form['query']
        result = query.process_query(srch)
        if result:
            with open('result.csv', 'w', newline='', encoding='utf-8') as file:
                datawriter = csv.DictWriter(file, delimiter='\t',
                                            fieldnames=['text_id', 'sentence', 'sentiment', 'film_name'])
                for sentence in result:
                    datawriter.writerow(sentence)
            return redirect(url_for('search_result'))
        else:
            return redirect(url_for('nothing_found'))
    return render_template('search.html')


@app.route('/search_result', methods=['GET', 'POST'])
def search_result():
    if request.method == 'POST':
        return send_file("result.csv", as_attachment=True, attachment_filename="corpora_result.csv")
    df = pd.read_csv('result.csv', delimiter='\t', encoding='utf-8')
    out = df.values.tolist()
    out_ = out
    return render_template('search_result.html', out_=out_)


@app.route('/nothing_found', methods=['GET', 'POST'])
def nothing_found():
    return render_template('nothing_found.html')


if __name__ == '__main__':
    app.run(debug=True)
