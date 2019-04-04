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

def review_id_tokens(yelp_review_fp: Path, 
                     tokeniser: Callable[[str], List[str]],
                     lower: bool
                     ) -> Iterable[Tuple[str, List[str]]]:
    '''
    Given a file path to the Yelp review data, it will lower case and tokenise 
    the text and generate for each review the ID and tokens of the review.
    '''
    for _id, review in review_id_data(yelp_review_fp):
        text = review['text']
        if lower:
            text = text.lower()
        tokens = tokeniser(text)
        yield _id, tokens

def write_tokens(data_fp: Path, data_ids: Set[str], yelp_review_fp: Path, 
                 tokeniser: Callable[[str], List[str]], lower: bool) -> int:

    count = 0
    total_tokens = 0
    with data_fp.open('w+') as data_file:
        for _id, tokens in review_id_tokens(yelp_review_fp, tokeniser, lower):
            if _id in data_ids:
                text = ' '.join(tokens)
                if count != 0:
                    text = f"\n{text}"
                data_file.write(text)

                count += 1
                total_tokens += len(tokens)
    return total_tokens

#def write_data(data_fp: Path, tokens: List[str]) -> None:
#    '''
#    Given a File path will join the tokens back up based on whitespace and 
#    append the string to the given File Path.
#
#    The File path should be the file path to either the training, validation, 
#    or test file. 
#    '''
#    text = ' '.join(tokens)
#    if data_fp.exists():
#        text = f"\n{text}"
#    with data_fp.open('a+') as data_file:
#        data_file.write(text)


if __name__ == '__main__':
    '''
    The tokeniser used here is the whitespace tokeniser.
    '''

    split_ids_help = 'File Path to a json file that contains three keys: '\
                     '`train`, `val`, `test` each having values that are lists'\
                     ' of review ids associated to the yelp review dataset'
    rewrite_help = "Whether or not to rewrite over the data stored in the "\
                   "train, validation, and test file paths"

    parser = argparse.ArgumentParser()
    parser.add_argument("review_data", type=parse_path,
                        help='File path to the Yelp review data.')
    parser.add_argument("-split_ids", type=parse_path, help=split_ids_help)
    parser.add_argument("--rewrite", help=rewrite_help, action="store_true")
    parser.add_argument("--lower", help='Lower case the words', 
                        action="store_true")
    parser.add_argument("train_fp", type=parse_path, 
                        help="File Path to store the train data")
    parser.add_argument("val_fp", type=parse_path, 
                        help="File Path to store the validation data")
    parser.add_argument("test_fp", type=parse_path, 
                        help="File Path to store the test data")
    args = parser.parse_args()

    split_ids_fp = args.split_ids
    if args.split_ids is None:
        split_ids_fp = Path(__file__, '..', 'split_ids.json').resolve()
    else:
        if not split_ids_fp.exists():
            raise ValueError(f'The split ids file has to exist: {split_ids_fp}')

    expected_num_reviews = 6685900

    yelp_review_data_fp = args.review_data
    # If this is True then we need to create the split review ids, else we load
    # them from the json file
    train_ids, val_ids, test_ids = list(), list(), list()
    if not split_ids_fp.exists():
        test_val_num_reviews = math.ceil(expected_num_reviews * 0.08)
        review_counts = set([i for i in range(expected_num_reviews)])

        test_counts = set(random.sample(review_counts, test_val_num_reviews))
        review_counts = review_counts.difference(test_counts)
        val_counts = set(random.sample(review_counts, test_val_num_reviews))
        train_counts = review_counts.difference(val_counts)

        assert len(test_counts.intersection(val_counts)) == 0
        assert len(train_counts.intersection(val_counts)) == 0
        assert len(train_counts.intersection(test_counts)) == 0

        count = 0
        for _id, review in review_id_data(yelp_review_data_fp):
            
            if count in train_counts:
                train_ids.append(_id)
            elif count in val_counts:
                val_ids.append(_id)
            elif count in test_counts:
                test_ids.append(_id)
            else:
                raise ValueError(f'This count {count} is not in any of the '
                                 'counts, largest possible count value '
                                 f'{expected_num_reviews - 1}')
            count += 1
        with split_ids_fp.open('w+') as split_ids_file:
            json.dump({'train': train_ids, 'val': val_ids, 'test': test_ids}, 
                      split_ids_file)
    else:
        with split_ids_fp.open('r') as split_ids_file:
            split_ids = json.load(split_ids_file)
            train_ids = split_ids['train']
            val_ids = split_ids['val']
            test_ids = split_ids['test']
    
    num_train_reviews = len(train_ids)
    num_val_reviews = len(val_ids)
    num_test_reviews = len(test_ids)
    print(f'Number of training reviews {num_train_reviews}'
          f'({num_train_reviews/expected_num_reviews}%)')
    print(f'Number of validation reviews {num_val_reviews}'
          f'({num_val_reviews/expected_num_reviews}%)')
    print(f'Number of test reviews {num_test_reviews}'
          f'({num_test_reviews/expected_num_reviews}%)')
    
    train_fp, val_fp, test_fp = args.train_fp, args.val_fp, args.test_fp
    will_rewrite = True
    for data_fp in [train_fp, val_fp, test_fp]:
        if data_fp.exists():
            will_rewrite = False
    if will_rewrite or args.rewrite:
        print('Creating training, validation, and test sets')

        review_id_tokens_params = [yelp_review_data_fp, str.split, args.lower]
        train_token_count = write_tokens(train_fp, train_ids, 
                                         *review_id_tokens_params)
        val_token_count = write_tokens(val_fp, val_ids, 
                                       *review_id_tokens_params)
        test_token_count = write_tokens(test_fp, test_ids, 
                                        *review_id_tokens_params)
        print(f'Token count for:\nTraining data {train_token_count}\n'
              f'Validation data {val_token_count}\nTest data {test_token_count}')

        #for _id, tokens in review_id_tokens(yelp_review_data_fp, 
        #                                    str.split, args.lower):
        #    num_tokens = len(tokens)
        #    if _id in train_ids:
        #        write_data(train_fp, tokens)
        #        train_token_count += num_tokens
        #    elif _id in val_ids:
        #        write_data(val_fp, tokens)
        #        val_token_count += num_tokens
        #    elif _id in test_ids:
        #        write_data(test_fp, tokens)
        #        test_token_count += num_tokens
        #    else:
        #        raise ValueError(f'id {_id} does not exist in any of the train,'
        #                         ' val, or test id lists')
        
    else:
        print('NOT CREATING DATA SETS as one of the files contains data and '
              '`rewrite argument has not been set`')
    
