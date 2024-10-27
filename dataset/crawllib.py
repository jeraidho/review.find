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
from typing import Optional, Callable, Tuple, Iterable, Any, Union, Literal
from pydantic import BaseModel, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema


# class for Review with typing
class Review(BaseModel):
    id: int
    film_name: str
    sentiment: Literal['negative', 'positive', 'neutral']
    text: str
    tokens: Optional[list[Tuple]]


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
        self.url = 'https://www.kinopoisk.ru/reviews/type/comment/period/month/perpage/50/'

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
                    print('Captcha got, reconnect')
                    time.sleep(random.uniform(5, 12))
                    req = self.session.get(page_url, headers={'User-Agent': self.user_agent.random}, timeout=10)
            except Exception as e:
                self.errors.append(e)
            else:
                soup = BeautifulSoup(req.text, 'html.parser')

                # extract all reviews
                reviews = soup.find_all('div', {'class': 'reviewItem userReview'})
                print(len(reviews))

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
            json.dump({review.id: review.json() for review in self.reviews}, f)


def load_reviews(self, filename: str):
    """
    Load in local var all reviews to work with them
    :param filename: name of file with reviews
    :return: class instance
    """
    with open(filename, 'rb') as f:
        self.reviews = pickle.load(f)
    return self


if __name__ == "__main__":
    crawler = ReviewCrawler()
    crawler.crawl(json_autosave=True)
    crawler.pickle_save()
