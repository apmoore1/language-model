import json
from pathlib import Path
from typing import Iterable, Callable

def yelp_text_generator(yelp_fp: Path) -> Iterable[str]:
    with yelp_fp.open('r') as yelp_file:
        for line in yelp_file:
            line = line.strip()
            if line:
                review = json.loads(line)
                review_text = review['text']
                yield review_text

def amazon_text_generator(amazon_fp: Path) -> Iterable[str]:
    with amazon_fp.open('r') as amazon_file:
        for line in amazon_file:
            line = line.strip()
            if line:
                review = json.loads(line)
                review_text = review['reviewText']
                yield review_text

def sentence_text_generator(data_fp: Path) -> Iterable[str]:
    with data_fp.open('r') as data_file:
        for line in data_file:
            line = line.strip()
            if line:
                yield line

def text_generator_func_mapper(review_data_name: str) -> Callable[[Path], Iterable[str]]:
    if review_data_name == 'yelp':
        return yelp_text_generator
    elif review_data_name == 'yelp_sentences' or \
         review_data_name == 'billion_word_corpus':
        return sentence_text_generator
    elif review_data_name == 'amazon':
        return amazon_text_generator
    else:
        raise ValueError('Do not have a text generator function for the '
                         f'{review_data_name} review data')