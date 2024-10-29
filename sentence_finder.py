import json
import re


class SentenceFinder:
    def __init__(self, metadata, tokendata, lemma_index_data):
        self.metadata = metadata
        self.tokendata = tokendata
        self.lemma_index_data = lemma_index_data

    def load_data(self, metadata_path, tokendata_path, lemma_index_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(tokendata_path, 'r', encoding='utf-8') as f:
            tokendata = json.load(f)
        with open(lemma_index_path, 'r', encoding='utf-8') as f:
            lemma_index_data = json.load(f)
        return metadata, tokendata, lemma_index_data

    def find_sentences_with_token(self, token, metadata, tokendata):
        token_results = []

        if tokendata[token]:
            entry = tokendata[token]

            text_sentence_id = entry['text_sentence_id']
            upos = entry['upos']
            features = entry['features']
            deprel = entry['deprel']
            # print(text_sentence_id)
            for i in range(len(text_sentence_id)):
                # for t_s_id in text_sentence_id:
                t_s_id = text_sentence_id[i]
                text_id, sentence_index, word_index = t_s_id.split('_')
                # print(text_id)
                upo = upos[i]
                feature = features[i]
                depr = deprel[i]
                if text_id in metadata and 'sentences' in metadata[text_id]:
                    sentences = metadata[text_id]['sentences']
                    sentiment = metadata[text_id]['sentiment']
                    film_name = metadata[text_id]['film_name']

                    if int(sentence_index) < len(sentences):
                        results = {}
                        sentence = sentences[int(sentence_index)]
                        results["text_id"] = text_id
                        results["sentence"] = sentence
                        results["sentence_index"] = sentence_index
                        results["word_index"] = word_index
                        results["upos"] = upo
                        results["features"] = feature
                        results["deprel"] = depr
                        results["sentiment"] = sentiment
                        results["film_name"] = film_name
                token_results.append(results)

        # print(token_results)
        return token_results

    # поиск фичи c токеном
    def search_feature(self, feature, feature_value, token_results):
        have_feature = []
        for result in token_results:

            if result[feature] == feature_value:

                have_feature.append(result)
            else:
                if feature == "features" and feature_value in result[feature]:  # проверить не будет ли конфуза
                    have_feature.append(result)
        return have_feature

        # find lemma

    def find_sentences_with_lemma(self, lemma, metadata, tokendata, lemma_index_data):
        lemma_results = []
        for token in lemma_index_data[lemma]:
            lemma_results.append(self.find_sentences_with_token(token, metadata, tokendata))
        return lemma_results

    # ПОИСК ФИЧИ БЕЗ ТОКЕНА

    def find_sentences_with_feature_no_token(self, feature_query, metadata, tokendata):

        results = []

        # Разделяем запрос на ключ и значение
        feature_key, feature_value = feature_query.split('=')

        for token in tokendata:

            # print(token_key, "#", token_value)
            # print(tokendata[token])
            # print(len(tokendata[token]))
            # print(len(tokendata[token]["features"]))
            for i in range(len(tokendata[token][feature_key])):
                dic = {}
                elem = tokendata[token][feature_key][i]
                # print(elem)
                if elem != None:
                    if feature_value == elem or feature_key == "features" and feature_value in elem:
                        # Получаем text_id и sentence_id из text_sentence_id
                        text_sentence_id = tokendata[token]['text_sentence_id'][i]
                        # print(text_sentence_id)
                        text_id = text_sentence_id.split('_')[0]
                        sentence_id = int(text_sentence_id.split('_')[1])
                        word_index = text_sentence_id.split('_')[2]
                        sentiment = metadata[text_id]["sentiment"]
                        film_name = metadata[text_id]["film_name"]

                        # Получаем предложение из metadata
                        if text_id in metadata:
                            sentences = metadata[text_id]['sentences']
                            if 0 <= sentence_id < len(sentences):
                                sentence = sentences[sentence_id]
                                dic["text_id"] = text_id
                                dic["sentence"] = sentence
                                dic['word_index'] = word_index
                                dic['sentiment'] = sentiment
                                dic['film_name'] = film_name
                                results.append(dic)

        return results

    def has_cyrillic(self, text):
        return bool(re.search('[а-яА-Я]', text))

    def understand_query_part(self, query_part):
        if "'" in query_part:
            if "+" in query_part:
                return "token+tag"
            else:
                return "token"
        elif self.has_cyrillic(query_part):
            if "+" in query_part:
                return "lemma+tag"
            else:
                return "lemma"
        else:
            return "tag"

    def get_consecutive_sentences(self, final_dict):
        # Словарь для хранения предложений по text_id
        sentences = {}

        # Проходим по каждому слову в словаре
        for word, entries in final_dict.items():
            for entry in entries:
                text_id = entry['text_id']
                sentence = entry['sentence']
                word_id = int(entry['word_index'])
                sentiment = entry['sentiment']
                film_name = entry['film_name']

                # Создаем ключ для предложения
                key = (text_id, sentence)

                if key not in sentences:
                    sentences[key] = []

                # Добавляем word_id в список для данного предложения
                sentences[key].append(word_id)

        result = []

        # Проверяем каждое предложение на наличие последовательных word_id
        for (text_id, sentence), word_ids in sentences.items():
            if len(word_ids) == len(final_dict):  # Проверяем, что предложение встречается у всех слов
                if sorted(word_ids) == list(
                        range(min(word_ids), min(word_ids) + len(word_ids))):  # Проверяем последовательность
                    #  print(list(range(min(word_ids), min(word_ids) + len(word_ids))))
                    result.append(
                        {'text_id': text_id, 'sentence': sentence, 'sentiment': sentiment, 'film_name': film_name})

        return result

    def process_query(self, query):
        query_list = query.split()
        final_dict = {}
        for word in query_list:
            if self.understand_query_part(word) == "token":
                final_dict[word] = self.find_sentences_with_token(word[1:-1], metadata, tokendata)
            if self.understand_query_part(word) == "lemma":
                lemma_results = self.find_sentences_with_lemma(word, metadata, tokendata, lemma_index_data)
                final = []
                for lemma in lemma_results:
                    final = final + lemma
                final_dict[word] = final
            # final_dict[word] = find_sentences_with_lemma(word, metadata, tokendata, lemma_index_data)

            if self.understand_query_part(word) == "token+tag":
                # предположим что пользователь должен вбить тип тэга = тэг в query
                token, tag = word.split("+")
                feature, feature_value = tag.split("=")
                token_results = self.find_sentences_with_token(token[1:-1], metadata, tokendata)
                final_dict[word] = self.search_feature(feature, feature_value, token_results)
            if self.understand_query_part(word) == "lemma+tag":
                lemma, tag = word.split("+")
                feature, feature_value = tag.split("=")
                lemma_results = self.find_sentences_with_lemma(lemma, metadata, tokendata, lemma_index_data)
                test_lemma_feature = []
                for token in lemma_results:
                    token_feature = self.search_feature(feature, feature_value, token)
                    test_lemma_feature.append(token_feature)
                final = []
                for lemma in test_lemma_feature:
                    final = final + lemma
                    final_dict[word] = final
            if self.understand_query_part(word) == "tag":
                final_dict[word] = self.find_sentences_with_feature_no_token(feature_query, metadata, tokendata)

        # не забыть добавить для просто тэга
        # common_sentences = get_common_sentences(final_dict, query_list)
        consecutive_sentences = self.get_consecutive_sentences(final_dict)

        return consecutive_sentences


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Пример использования класса
if __name__ == "__main__":
    metadata = load_json_data('metadata.json')
    tokendata = load_json_data('tokendata.json')
    lemma_index_data = load_json_data('lemma_index.json')

    finder = SentenceFinder(metadata, tokendata, lemma_index_data)

    # Запрос пользователя
    user_query = input("Введите ваш запрос (например, token=lemma1): ")

    # Обработка запроса
    found_sentences = finder.process_query(user_query)

    # Вывод результатов
    if found_sentences:
        print(found_sentences)
    else:
        print("Предложения не найдены.")