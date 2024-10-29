from flask import Flask
from flask import url_for, render_template, request, redirect
import csv
import pandas as pd
from dataset/datalib import load_json_data
from dataset/datalib import SentenceFinder
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # главная страница

@app.route('/search', methods = ['GET', 'POST'])
def query_process():
    metadata = load_json_data('metadata.json')
    tokendata = load_json_data('tokendata.json')
    lemma_index_data = load_json_data('lemma_index.json')
    if request.method == 'POST':
        search = request.form['query']
        query = SentenceFinder(metadata, tokendata, lemma_index_data)
        result = query.process_query(search)
        with open('result.csv', 'w', newline='', encoding='utf-8') as file:
            datawriter = csv.DictWriter(file, delimiter='\t', fieldnames=['text_id', 'sentence', 'sentiment', 'film_name'])
             for sentence in result:
                datawriter.writerow(sentence)
        return redirect(url_for('search_result'))
    return render_template('search.html')

@app.route('/search_result')
def output_page():
    df = pd.read_csv('result.csv', encoding='utf-8')
    out = df.values.tolist()
    out_ = out
    return render_template('search_result.html', out_=out_)

if __name__ == '__main__':
    app.run(debug=True)