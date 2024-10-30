from flask import Flask
from flask import url_for, render_template, request, redirect
import csv
import pandas as pd
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')  # главная страница


@app.route('/search')
def render_search_page():
    return render_template('search.html') # страница поиска


@app.route('/search_res_test')
def render_result_page():
    return render_template('search_result.html', out_=[["001", "Ужасно снятый фильм.", "negative", "Ёлки"],
                                                       ["001", "Ужасно увлекательный фильм!", "positive", "Палки"],
                                                       ["001", "Ужасно весело!", "positive", "Ёлки"]]) # страница поиска


# @app.route('/search', methods = ['GET', 'POST'])
# def query_process():
#     if request.method == 'POST':
#         search = request.form['query']
#         query = Finder()
#         metadata, tokendata, lemma_index_data = query.load_data('metadata.json', 'tokendata.json', "lemma_index.json")
#         result = query.process_query(search, metadata, tokendata, lemma_index_data)
#         with open('result.csv', 'w', newline='', encoding='utf-8') as file:
#             datawriter = csv.DictWriter(file, delimiter='\t', fieldnames=['text_id', 'sentence', 'sentiment', 'film_name'])
#             for sentence in result:
#                 datawriter.writerow(sentence)
#         return redirect(url_for('search_result'))
#     return render_template('search.html')
#
# def output_page():
#     df = pd.read_csv('result.csv', encoding='utf-8')
#     out = df.values.tolist()
#     out_ = out
#     return render_template('search_result.html', out_=out_)

if __name__ == '__main__':
    app.run(debug=True)