import argparse
from pathlib import Path
import json


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    '''
    Filter by business categories. This script will create a json file that 
    will store a list of business IDS where these business are all within the 
    categories specified as an argument.

    Within our work as we only want reviews from restaurants the categories we 
    used are the following where this is converted to a list via whitespace 
    splitting:
    
    'restaurants restaurant restaurants,'
    '''
    filtered_fp_help = 'Path to the file that will store all the filtered '\
                       'business IDS'
    cat_help = 'A string containing the category names seperated by whitespace'
    parser = argparse.ArgumentParser()
    parser.add_argument("business_fp", type=parse_path, 
                        help='File path to the Yelp business json file')
    parser.add_argument("filtered_business_ids_fp", type=parse_path, 
                        help=filtered_fp_help)
    parser.add_argument("categories", type=str, 
                        help=cat_help)
    args = parser.parse_args()

    filter_categories = set(args.categories.lower().split())
    print(f'Filtered Categories: {filter_categories}')

    total_count = 0
    filtered_business_ids = set()

    with args.business_fp.open('r') as bis_file:
        for line in bis_file:
            line = json.loads(line)
            if 'categories' in line:
                if isinstance(line['categories'], str):
                    line_categories = line['categories'].lower().split()
                    for cat in line_categories:
                        if cat in filter_categories:
                            filtered_business_ids.add(line['business_id'])
            total_count += 1
    
    print(f'Total number of business: {total_count}')
    print('Number of business within the filtered list: '
          f'{len(filtered_business_ids)}')
    
    with args.filtered_business_ids_fp.open('w+') as business_id_file:
        filtered_business_ids = list(filtered_business_ids)
        json.dump(filtered_business_ids, business_id_file)