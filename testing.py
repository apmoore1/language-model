from pathlib import Path

from allennlp.nn import InitializerApplicator, Initializer
from allennlp.common import from_params, Params
from allennlp.data import Vocabulary, DatasetReader
from allennlp.models import Model, load_archive


transformer_weights_fp = str(Path('/home/andrew/Downloads/transformer_unpacked/data/weights.th').resolve())
transformer_model_fp = str(Path('/home/andrew/Downloads/transformer-elmo-2019.01.10.tar.gz').resolve())
yelp_model_config = str(Path('/home/andrew/Downloads/yelp_lm.json').resolve())

model = load_archive(transformer_model_fp)
named_mods = [(path, module) for path, module in model.model.named_modules()]
names = [name for name, mod in named_mods]
mods = [mod for name, mod in named_mods]
char_cnn_0 = list(mods[2].parameters())[0]
names_mods = list(zip(names, mods))
con_name_mod = [(name,mod) for name, mod in names_mods if name=='_contextualizer']
con_params = list(con_name_mod[0][1].parameters())
con_params_1 = con_params[0]

params = Params.from_file(yelp_model_config)
#yelp_init_params = params['initializer']
#pre_init_name = yelp_init_params[0][0]
#pre_init = Initializer.from_params(params['initializer'][0][1])
#yelp_init = InitializerApplicator([(pre_init_name, pre_init)])

reader = DatasetReader.from_params(params['dataset_reader']['base_reader'])
instances = reader.read(params['train_data_path'])
if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
else:
    raise ValueError('something')

yelp_model = Model.from_params(vocab=vocab, params=params['model'])
#yelp_init(yelp_model)
named_mods_yelp = [(path, module) for path, module in yelp_model.named_modules()]
names_yelp = [name for name, mod in named_mods_yelp]
mods_yelp = [mod for name, mod in named_mods_yelp]
char_cnn_yelp_0 = list(mods_yelp[2].parameters())[0]
names_mods_yelp = list(zip(names_yelp, mods_yelp))
con_name_mod_yelp = [(name,mod) for name, mod in names_mods_yelp if name=='_contextualizer']
con_params_yelp = list(con_name_mod_yelp[0][1].parameters())
con_params_1_yelp = con_params_yelp[0]
print('done')