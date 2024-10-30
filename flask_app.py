from flask import Flask
from flask import url_for, render_template, request, redirect
import csv
import pandas as pd
from dataset.datalib import load_json_data
from dataset.datalib import SentenceFinder
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
        if result:
            with open('result.csv', 'w', newline='', encoding='utf-8') as file:
                datawriter = csv.DictWriter(file, delimiter='\t', fieldnames=['text_id', 'sentence', 'sentiment', 'film_name'])
                for sentence in result:
                    datawriter.writerow(sentence)
            return redirect(url_for('search_result'))
        else:
            with open('result.txt', 'w', encoding='utf-8') as file:
                file.write('Предложения не найдены')
            return redirect(url_for('nothing_found'))
    return render_template('search.html')

@app.route('/search_result')
def output_page():
    df = pd.read_csv('result.csv', encoding='utf-8')
    out = df.values.tolist()
    out_ = out
    return render_template('search_result.html', out_=out_)

@app.route('/nothing_found')
def nothing_found():
    with open('result.txt', 'r', encoding='utf-8') as file:
        nothing = file.read()
    out_ = nothing
    return render_template('nothing_found.html', out_=out_)

if __name__ == '__main__':
    app.run(debug=True)