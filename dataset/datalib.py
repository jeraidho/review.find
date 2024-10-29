import requests
from fake_useragent import UserAgent
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
import json
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pickle
from typing import Literal
from pydantic import BaseModel
import stanza
import warnings
from collections import defaultdict


# class for Review with typing
class Review(BaseModel):
    id: int
    film_name: str
    sentiment: Literal['negative', 'positive', 'neutral']
    text: str


class ReviewCrawler:

    def __init__(self):
        # vars for connection
        self.session = requests.session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.user_agent = UserAgent()

        # main url
        self.url = 'https://www.kinopoisk.ru/reviews/type/comment/period/month/page/'

        # filename and time for backup while crawling
        self.moment = time.strftime("%Y-%b-%d_%H_%M_%S", time.localtime())
        self.review_backup_name = 'dataset.json'

        # inner vars for mined data
        self.reviews = []
        self.errors = []

        # counters for collected reviews and saved to json
        self.counter = 0
        self.saved = 0

        # service converter for values
        self.sentiment_converter = {'response': 'neutral',
                                    'response bad': 'bad',
                                    'response good': 'good'}

    def extract_page_number(self) -> int:
        """
        Service function to get last page of reviews
        :return: number of last page
        """
        # get first page
        req = self.session.get(self.url + '1/#list', headers={'User-Agent': self.user_agent.random}, timeout=10)
        while 'captcha' in req.text:
            print('Captcha got, reconnect...')
            time.sleep(random.uniform(5, 12))
            req = self.session.get(self.url + '1/#list', headers={'User-Agent': self.user_agent.random}, timeout=10)
        soup = BeautifulSoup(req.text, 'html.parser')
        return int(soup.find_all('li', {'class': 'arr'})[-1].find('a',
                                                                  string='»»').attrs['href'].split('/')[-2])

    def crawl(self, json_autosave: bool = False) -> None:
        """
        Get hrefs of full reviews in backup file and in class var
        :param json_autosave: if True saves every 5 reviews in json
        :return: List or None
        """
        # not saved href counter and total review counter

        for page in tqdm(range(1, self.extract_page_number() + 1),
                         colour='green', desc=f'Crawl reviews'):
            time.sleep(random.uniform(1, 4))

            page_url = self.url + str(page) + '/#list'

            # request page and get reviews from it
            try:
                req = self.session.get(page_url, headers={'User-Agent': self.user_agent.random}, timeout=10)
                while 'captcha' in req.text:
                    print(f'Captcha got on page {page}, reconnect')
                    time.sleep(random.uniform(5, 12))
                    req = self.session.get(page_url, headers={'User-Agent': self.user_agent.random}, timeout=10)
            except Exception as e:
                self.errors.append(e)
            else:
                soup = BeautifulSoup(req.text, 'html.parser')

                # extract all reviews
                reviews = soup.find_all('div', {'class': 'reviewItem userReview'})

                # insert info to class Review for each class
                for review in tqdm(reviews, colour='red', desc='Reviews on page'):
                    # get data
                    id = self.counter
                    text = review.find('span', {'itemprop': 'reviewBody'}).text
                    sentiment = review.find('div',
                                            class_=lambda value: value
                                                                 and value.startswith('response')).attrs['class'][0]
                    film_name = review.find('p', class_='film').find('span', {'itemprop': 'name'}).text

                    # add instance of review
                    self.reviews.append(Review(id=id, film_name=film_name,
                                               sentiment=self.sentiment_converter[sentiment], text=text))

                    self.counter += 1

                    # save if autosave True and unsaved more than 5
                    if json_autosave and self.counter - self.saved == 20:
                        self.json_save()
                        self.saved = self.counter

    def pickle_save(self):
        with open('reviews-' + self.moment + '.pkl', 'wb') as f:
            pickle.dump(self.reviews, f)

    def json_save(self):
        with open('reviews-' + self.moment + '.json', 'w') as f:
            json.dump({review.id: review.dict() for review in self.reviews}, f)


# load reviews from binary file
def load_reviews(self, filename: str):
    """
    Load in local var all reviews to work with them
    :param filename: name of file with reviews
    :return: class instance
    """
    with open(filename, 'rb') as f:
        self.reviews = pickle.load(f)
    return self


# class for different layers of dataset
# we will have two json files: metadata.json (id: index of review, film_name: name of review film,
# sentiment: meta information about review (negative|positive|neutral), text: full text of review,
# sentences: list of sentences in review)
# other tokendata.json (token: {text_id, sentence_id, lemma, upos, role}) contains token from text, id of text
# and id of sentence in text, part of speech tag,

class Processor:

    def __init__(self, reviews: dict):
        self.reviews = reviews

        # stanza initiation
        stanza.download('ru')
        self.nlp = stanza.Pipeline('ru')

        # lists for main data and token data (see description above)
        self.metadata = {}
        self.tokendata = {}

        # dict for already created indexes
        self.indexes = {}

        self.metadata_filename = 'metadata.json'
        self.tokens_filename = 'tokendata.json'

    def render_meta(self, file_output: bool = False):
        """
        Creates metadata file structure and if needed file metadata.json with dataset of reviews
        :param file_output: creates json file if True
        :return: None
        """
        # add data -- sentences
        self.metadata = {key: {
            'film_name': value['film_name'],
            'sentiment': value['sentiment'],
            'text': value['text'],
            'sentences': [sentence.text for sentence in self.nlp(value['text']).sentences]
        } for key, value in tqdm(self.reviews.items(), desc='Render reviews')}

        # create file with metadata
        if file_output:
            with open(self.metadata_filename, 'w') as f:
                json.dump(self.metadata, f)

    def render_tokens(self, file_output: bool = False):
        """
        Create tokens data structure and creates file tokendata.json with dataset if needed
        :param file_output: creates json file if True
        :return: None
        """
        # check if metadata inserted
        if not self.metadata:
            raise ValueError('Metadata is not rendered, consider using render_meta method')

        # iterate through reviews
        for key, value in tqdm(self.metadata.items(), desc='Rendering tokens in reviews'):
            for index, sentence in enumerate(value['sentences']):
                for token in self.nlp(sentence).sentences[0].words:
                    lower_token = token.text.lower()
                    if lower_token not in self.tokendata:
                        self.tokendata[lower_token] = {
                            'text_sentence_id': [f'{key}_{index}_{token.id}'],
                            'lemma': [token.lemma],
                            'upos': [token.upos],
                            'features': [token.feats],
                            'deprel': [token.deprel]
                        }
                    else:
                        self.tokendata[lower_token]['text_sentence_id'].append(f'{key}_{index}_{token.id}')
                        self.tokendata[lower_token]['lemma'].append(token.lemma)
                        self.tokendata[lower_token]['upos'].append(token.upos)
                        self.tokendata[lower_token]['features'].append(token.feats)
                        self.tokendata[lower_token]['deprel'].append(token.deprel)

        # create file with
        if file_output:
            with open(self.tokens_filename, 'w') as f:
                json.dump(self.tokendata, f)

    def create_index(self, feature: str, file_output: bool = False):
        """
        Create index and file feature_index.json with index on chosen feature from tokendata
        :param feature: feature of token from tokendata
        :param file_output: creates json file if True
        :return: None
        """

        # check if tokendata is inserted
        if not self.tokendata:
            raise ValueError('Tokendata is not rendered, consider using render_tokens method')

        if feature in self.indexes:
            warnings.warn(f'Index for {feature} already created, it is going to be overwritten')

        # check if token feature is correct
        if feature not in ['text_sentence_id', 'lemma', 'upos', 'features', 'deprel']:
            raise ValueError(f'Feature "{feature}" is incorrect')

        # result data
        index = defaultdict(list)

        # iterate through tokens
        for key, value in tqdm(self.tokendata.items(), desc=f'Create index for "{feature}"'):
            for item in value[feature]:
                if key not in index[item]:
                    index[item].append(key)

        # add created index in indexes
        self.indexes[feature] = index

        # create file with metadata
        if file_output:
            with open(feature + '_index.json', 'w') as f:
                json.dump(index, f)

    def create_inner_index(self, feature1: str, feature2: str, file_output: bool = False):
        """
        Create index feature-to-feature and file feature_index.json with index on chosen feature from tokendata
        :param feature1: feature of token from tokendata
        :param feature2: feature of token from tokendata
        :param file_output: creates json file if True
        :return: None
        """
        # check if tokendata is inserted
        if not self.tokendata:
            raise ValueError('Tokendata is not rendered, consider using render_tokens method')

        if feature1 + '_' + feature2 in self.indexes:
            warnings.warn(f'Index for this features already created, it is going to be overwritten')

        if feature1 == feature2:
            raise ValueError('Equal features are inserted')

        for feature in [feature1, feature2]:
            # check if token feature is correct
            if feature not in ['text_sentence_id', 'lemma', 'upos', 'features', 'deprel']:
                raise ValueError(f'Feature "{feature}" is incorrect')

        # result data
        index = defaultdict(list)

        # iterate through tokens
        for key, value in tqdm(self.tokendata.items(), desc=f'Create index for "{feature1}" and "{feature2}"'):
            for item in zip(value[feature1], value[feature2]):
                if item[1] not in index[item[0]]:
                    index[item[0]].append(item[1])

        # add created index in indexes
        self.indexes[feature1 + '_' + feature2] = index

        # create file with metadata
        if file_output:
            with open(feature1 + '_' + feature2 + '_index.json', 'w') as f:
                json.dump(index, f)

    def read_data(self, type: str, filename: str = '', data: dict = {}):

        """
        Save data (tokendata or metadata) to class instance from file or from data
        :param type: metadata or tokendata
        :param filename: json filename with data
        :param data: data itself in format of file
        :return: None
        """

        if type not in ['metadata', 'tokendata']:
            raise ValueError('Incorrect data inserted')

        # if data already inserted
        if (type == 'metadata' and self.metadata) or (type == 'tokendata' and self.tokendata):
            warnings.warn('Warning: data is already inserted, this method is going to override it')
        # some warnings
        # if no data provided
        if not filename and not data:
            raise ValueError('No data provided')

        # if both ways are inserted
        if filename and data:
            raise ValueError('Please choose only one way to upload data: from file or directly')

        # save data in self.metadata
        if filename and not data:
            with open(filename, 'r') as f:
                if type == 'metadata':
                    self.metadata = json.load(f)
                if type == 'tokendata':
                    self.tokendata = json.load(f)
                elif not filename and data:
                    if type == 'metadata':
                        self.metadata = data
                    if type == 'tokendata':
                        self.tokendata = data


if __name__ == "__main__":
    # collect dataset
    crawler = ReviewCrawler()
    crawler.crawl(json_autosave=True)
    crawler.pickle_save()

    # create two json-files of dataset
    with open('reviews.json', 'r') as f:
        reviews = json.load(f)

    processor = Processor(reviews)
    processor.render_meta(file_output=True)
    # processor.read_data('metadata', 'metadata.json')
    processor.render_tokens(file_output=True)
    # processor.read_data('tokendata', 'tokendata.json')
    processor.create_index('lemma', file_output=True)
    processor.create_inner_index('upos', 'text_sentence_id', file_output=True)
