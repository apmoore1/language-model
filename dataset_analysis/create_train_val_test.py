import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, List, Callable, Set
import random


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def review_id_data(yelp_review_fp: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    '''
    Given a file path to the Yelp review data, it will generate for each 
    review the ID and a dictionary of all the review data e.g. text, user_id, 
    stars etc.
    '''
    with yelp_review_fp.open('r') as review_data:
        for line in review_data:
            review = json.loads(line)
            review_id = review['review_id']
            yield review_id, review

#def review_id_tokens(yelp_review_fp: Path, 
#                     tokeniser: Callable[[str], List[str]],
#                     lower: bool
#                     ) -> Iterable[Tuple[str, List[str]]]:
#    '''
#    Given a file path to the Yelp review data, it will lower case and tokenise 
#    the text and generate for each review the ID and tokens of the review.
#    '''
#    for _id, review in review_id_data(yelp_review_fp):
#        text = review['text']
#        if lower:
#            text = text.lower()
#        tokens = tokeniser(text)
#        yield _id, tokens

#def write_tokens(data_fp: Path, data_ids: Set[str], yelp_review_fp: Path, 
#                 tokeniser: Callable[[str], List[str]], lower: bool) -> int:

#    count = 0
#    total_tokens = 0
#    with data_fp.open('w+') as data_file:
#        for _id, tokens in review_id_tokens(yelp_review_fp, tokeniser, lower):
#            if _id in data_ids:
#                text = ' '.join(tokens)
#                if count != 0:
#                    text = f"\n{text}"
#                data_file.write(text)

#                count += 1
#                total_tokens += len(tokens)
#    return total_tokens

def write_review(data_fp: Path, review: Dict[str, Any]) -> None:
    '''
    Given a File path will join the tokens back up based on whitespace and 
    append the string to the given File Path.

    The File path should be the file path to either the training, validation, 
    or test file. 
    '''
    review = json.dumps(review)
    if data_fp.exists():
        review = f"\n{review}"
    with data_fp.open('a+') as data_file:
        data_file.write(review)

def data_stats(num_train_reviews: int, num_val_reviews: int, 
               num_test_reviews: int) -> str:
    total_num_reviews = num_train_reviews + num_val_reviews + num_test_reviews

    train_statement = f'Number of training reviews {num_train_reviews}'\
                      f'({num_train_reviews/total_num_reviews}%)\n'
    val_statement = f'Number of validation reviews {num_val_reviews}'\
                    f'({num_val_reviews/total_num_reviews}%)\n'
    test_statement = f'Number of test reviews {num_test_reviews}'\
                     f'({num_test_reviews/total_num_reviews}%)'
    return train_statement + val_statement + test_statement

def do_data_files_exist(data_dir_fp: Path) -> bool:
    '''
    Given the directory which will store the Yelp review data splits if the
    splits already exist returns True else False.
    '''
    if not Path(data_dir_fp, 'train.json').exists():
        return False
    if not Path(data_dir_fp, 'val.json').exists():
        return False
    if not Path(data_dir_fp, 'test.json').exists():
        return False
    return True


if __name__ == '__main__':
    '''
    This will create train, validation, and test Yelp review data given the 
    yelp review json file and the directory to which the review splits will be 
    sotred in.

    The splits will be stored in that directory under the following file names:
    1. train.json
    2. val.json
    3. test.json

    If rewrite flag is given then the data will be rewritten but not the same 
    reviews will be written to each train, val, and test files as the reviews 
    are randomly chosen.

    This script also ensures that the train, validation, and test sets are 
    split with 84%, 8%, 8% reviews respectively from the whole dataset.

    Optional choices are the following:
    1. --min_token_count -- Minimum number of tokens that have to be in a 
       review. This is based on whitespace tokenisation. We set this to 3.      
    2. --filter_by_business_ids -- Whether or not to filter the reviews by 
       business ids. This expects a json file containing a list of business 
       IDS to filter by. This can be done using the 
       filter_businesses_by_category.py script
    '''

    rewrite_help = "Whether or not to rewrite over the data stored in the "\
                   "train, validation, and test file paths"
    data_dir_help = "Directory to store the train, validation, and test "\
                    "reviews."
    min_token_count_help = "Minimum number of tokens that have to be in a "\
                           "review. This is based on whitespace tokenisation. "\
                           "We set this to 3."
    bis_filter_help = 'Whether or not to filter the reviews by business ids.'\
                      ' This expects a json file containing a list of '\
                      'business IDS to filter by. This can be done using the '\
                      'filter_businesses_by_category.py script'

    parser = argparse.ArgumentParser()
    parser.add_argument("review_data", type=parse_path,
                        help='File path to the Yelp review data.')
    parser.add_argument("--rewrite", help=rewrite_help, action="store_true")
    parser.add_argument("data_dir", type=parse_path, 
                        help=data_dir_help)
    parser.add_argument("--min_token_count", help=min_token_count_help, 
                        type=int)
    parser.add_argument("--filter_by_business_ids", type=parse_path, 
                        help=bis_filter_help)
    args = parser.parse_args()

    data_dir_fp = args.data_dir
    data_dir_fp.mkdir(parents=True, exist_ok=True)
    data_splits_exist = do_data_files_exist(data_dir_fp)

    train_fp = Path(data_dir_fp, 'train.json')
    val_fp = Path(data_dir_fp, 'val.json')
    test_fp = Path(data_dir_fp, 'test.json')

    end_statement = 'Training, validation, and test data are at the following'\
                    f' paths {train_fp}, {val_fp}, {test_fp}'
    if data_splits_exist and not args.rewrite:
        print(end_statement + ' and have not been re-written')
    else:
        if not args.filter_by_business_ids:
            expected_num_reviews = 6685900
            test_val_num_reviews = math.ceil(expected_num_reviews * 0.08)
            review_counts = set([i for i in range(expected_num_reviews)])
        

            test_counts = set(random.sample(review_counts, test_val_num_reviews))
            review_counts = review_counts.difference(test_counts)
            val_counts = set(random.sample(review_counts, test_val_num_reviews))
            train_counts = review_counts.difference(val_counts)

            assert len(test_counts.intersection(val_counts)) == 0
            assert len(train_counts.intersection(val_counts)) == 0
            assert len(train_counts.intersection(test_counts)) == 0

            train_ids, val_ids, test_ids = list(), list(), list()

            min_token_count = args.min_token_count
            count = 0
            for _id, review in review_id_data(args.review_data):
                if min_token_count:
                    tokens = review['text'].split()
                    if len(tokens) < min_token_count:
                        continue
                if count in train_counts:
                    write_review(train_fp, review)
                    train_ids.append(_id)
                elif count in val_counts:
                    write_review(val_fp, review)
                    val_ids.append(_id)
                elif count in test_counts:
                    write_review(test_fp, review)
                    test_ids.append(_id)
                else:
                    raise ValueError(f'This count {count} is not in any of the '
                                    'counts, largest possible count value '
                                    f'{expected_num_reviews - 1}')
                count += 1
            
            
            print(data_stats(len(train_ids), len(val_ids), len(test_ids)))
        else:
            with args.filter_by_business_ids.open('r') as business_id_file:
                business_ids = set(json.load(business_id_file))

            train_ids, val_ids, test_ids = list(), list(), list()
            for _id, review in review_id_data(args.review_data):
                if review['business_id'] in business_ids:
                    split = random.choices([1,2,3], weights=[84,8,8])[0]
                    if split == 1:
                        write_review(train_fp, review)
                        train_ids.append(_id)
                    elif split == 2:
                        write_review(val_fp, review)
                        val_ids.append(_id)
                    elif split == 3:
                        write_review(test_fp, review)
                        test_ids.append(_id)

            print(data_stats(len(train_ids), len(val_ids), len(test_ids)))
        print(end_statement)
    
