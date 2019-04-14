from pathlib import Path
from typing import Callable, List
import re

from bella.tokenisers import spacy_tokeniser

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def get_tokeniser(tokeniser_name: str) -> Callable[[str], List[str]]:
    '''
    Given a tokeniser name it will return that tokeniser function.
    '''
    if tokeniser_name == 'spacy':
        print('Using the spaCy tokeniser')
        return spacy_tokeniser
    elif tokeniser_name == 'whitespace':
        print('Using the WhiteSpace tokeniser')
        return str.split
    else:
        raise ValueError(f'Not a recognised tokeniser name: {tokeniser_name}')

def valid_text_file_name_prefix(prefix_pattern: str, text_file_name: str
                                ) -> bool:
    if prefix_pattern:
        if re.search(rf'^{prefix_pattern}', text_file_name) is None:
            return False
    return True