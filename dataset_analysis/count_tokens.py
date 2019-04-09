import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, List, Callable, Set
import statistics

import seaborn as sns
import matplotlib.pyplot as plt


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

def calculate_review_stats(to_calculate: bool, 
                           review_token_count: List[int], 
                           tokens: List[str]) -> None:
    if not to_calculate:
        return
    review_token_count.append(len(tokens))


if __name__ == '__main__':
    '''
    Given a directory that contains three files:
    1. train.json
    2. val.json
    3. test.json

    And a name of the data source e.g. `yelp` it will count the number of 
    tokens that the total amount of text contains for each of these three 
    files. Where at the moment the only tokeniser that will be used is the 
    whitepsace tokeniser.

    Optional choices are the following:
    1. --lower -- if stated it will lower case all words
    2. --unique-count -- if stated it will print the unique number of tokens 
       for each file as well.
    3. --review_stats -- Prints the number of reviews, mean, max, min tokens
       per review for each of the dataset splits.
    4. --token_distribution -- Given a file path to a directory it will create 
       three files; train_token_dist.png, val_token_dist.png, 
       train_token_dist.png where each file will contain the token distribution 
       for reviews for each split.
    '''

    lower_help = "if stated it will lower case all words"
    unique_count_help = "if stated it will print the unique number of tokens "\
                        "for each file as well."
    data_dir_help = "Directory to where the train, validation, and test "\
                    "files are."
    review_stats_help = "Prints the number of reviews, mean, max, min token "\
                        "counts of reviews per split"

    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", action="store_true", help=lower_help)
    parser.add_argument("--unique_count", help=unique_count_help, 
                        action="store_true")
    parser.add_argument("--review_stats", help=review_stats_help, 
                        action="store_true")
    parser.add_argument("--token_distribution", type=parse_path, help='any')
    parser.add_argument("data_dir", type=parse_path, 
                        help=data_dir_help)
    parser.add_argument("data_source_name", type=str, choices=['yelp'], 
                        help='Name of the data source e.g. yelp')
    args = parser.parse_args()

    data_dir_fp = args.data_dir
    if not do_data_files_exist(data_dir_fp):
        raise FileNotFoundError('Cannot find one of the train, val, or test '
                                f'files in the following directory {data_dir_fp}')

    train_fp = Path(data_dir_fp, 'train.json')
    val_fp = Path(data_dir_fp, 'val.json')
    test_fp = Path(data_dir_fp, 'test.json')
    data_fps = [train_fp, val_fp, test_fp]

    data_source = args.data_source_name.lower().strip()
    to_lower = args.lower
    
    train_tokens = Counter()
    val_tokens = Counter()
    test_tokens = Counter()
    counters = [train_tokens, val_tokens, test_tokens]

    # If review level statistics are required
    create_token_distribution = args.token_distribution
    get_review_stats = args.review_stats
    split_names = ['train', 'validation', 'test']
    split_review_token_count = {split_name: [] for split_name in split_names}

    if data_source == 'yelp':
        for data_fp, token_counter, split_name in zip(data_fps, counters, split_names):
            for _id, tokens in review_id_tokens(data_fp, str.split, to_lower):
                token_counter.update(tokens)
                if get_review_stats or create_token_distribution:
                    split_review_token_count[split_name].append(len(tokens))

    for split_name, token_count in zip(split_names, counters):
        token_counts = sum(token_count.values())
        num_unique_tokens = len(token_count)
        print(f'{split_name}\nContains {token_counts} tokens')
        if args.unique_count:
            print(f'{num_unique_tokens} number of unique tokens')
        if get_review_stats:
           review_token_count = split_review_token_count[split_name]
           mean_count = statistics.mean(review_token_count)
           max_count = max(review_token_count)
           min_count = min(review_token_count)
           number_reviews = len(review_token_count)
           print(f'Number of reviews: {number_reviews}, Token count; max: '
                 f'{max_count}, mean: {mean_count}, min: {min_count}')
        if create_token_distribution:
            create_token_distribution.mkdir(parents=True, exist_ok=True)
            dist_fp = Path(create_token_distribution, f'{split_name}_token_dist.png')

            fig, ax = plt.subplots()
            ax = sns.distplot(split_review_token_count[split_name], ax=ax)
            fig.savefig(str(dist_fp.resolve()))

        print('\n')
    
