from pathlib import Path
from typing import Callable, List, Optional, Iterable
import re

from bella.data_types import TargetCollection
from bella.tokenisers import spacy_tokeniser
#import langid
#import numpy as np

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

def load_dataset_splits(directory: Path, dataset_name: str,
                        split_names: Optional[List[str]] = None
                        ) -> Iterable[TargetCollection]:
    split_names = split_names or ['Train', 'Val', 'Test']
    for split_name in split_names:
        data_split_fp = Path(directory, f'{dataset_name} {split_name}').resolve()
        if not data_split_fp.exists():
            raise FileNotFoundError('Cannot find the following TDSA data '
                                    f'split file {data_split_fp}')
        yield TargetCollection.load_from_json(data_split_fp)


#
# In the end we do not use Language identification to remove non-English 
# sentences as we found it not very good
#
#LANGID_MODEL = {}
#def _norm_probs(pd):
#    """
#    Renormalize log-probs into a proper distribution (sum 1)
#    The technique for dealing with underflow is described in
#    http://jblevins.org/log/log-sum-exp
#    """
    # Ignore overflow when computing the exponential. Large values
    # in the exp produce a result of inf, which does not affect
    # the correctness of the calculation (as 1/x->0 as x->inf). 
    # On Linux this does not actually trigger a warning, but on 
    # Windows this causes a RuntimeWarning, so we explicitly 
    # suppress it.
#    with np.errstate(over='ignore'):
#        pd_exp = np.exp(pd)
#        pd = pd_exp / pd_exp.sum()
#    return pd


#def _classify_language_id(text):
#    if 'model' not in LANGID_MODEL:
#        langid.langid.load_model()
#        langid.langid.identifier.norm_probs = _norm_probs
#        LANGID_MODEL['model'] = True
#    return langid.classify(text)

#def is_english(text: str, threshold: float = 0) -> bool:
#    lang, prob = _classify_language_id(text)
#    if lang == 'en' and prob > threshold:
#        return True
#    return False