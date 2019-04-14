# Difference between the 1 Billion test and training vocabularlies
First we need to create a vocabularly file (all the unique words) for both the training and test data from the 1 billion word corpus. To do this for the training data:
``` bash
python vocab_comparison/create_vocab.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../training_vocab.json whitespace --from_dir_of_files --file_prefix news
```
For the test data:
``` bash
python vocab_comparison/create_vocab.py ../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 ../test_vocab.json whitespace
```
The vocabularly files for training and test data are stored in `../training_vocab.json` and `../test_vocab.json` respectively. Now to see how many more words are in the testing vocabularly compared to the training we can run the following command:
``` bash
python vocab_comparison/comapre_vocabs.py ../test_vocab.json ../training_vocab.json ../diff_between_test_and_train.txt --not_symmetric
```
This should state that the difference is 13278 words/tokens and all of these words will be stored on a new line within `../diff_between_test_and_train.txt` text file. That is there are 13278 words in the test vocabularly that are not within the training vocabularly (this is not the same as the number of words that are within the training vocabularly but not within the test).

# Difference between the Transformer ELMo model's vocabularly and the 1 Billion training and testing Vocabularlies.
## Difference between Test and Transformer ELMo model's vocabularly
To find the differences first we need to transformer the model's vocabularly from a text file where each line contains a new word to a json file that stores them as a list as that is what is expected as input to the `vocab_comparison/comapre_vocabs.py` script and many other scripts. To do this run the following command:
``` bash
python vocab_comparison/txt_to_json.py ../vocab/vocab-2016-09-10.txt ../vocab/vocab-2016-09-10.json
```

Next to see how many of the model vocabularly words are not in the test vocabularly
``` bash
python vocab_comparison/comapre_vocabs.py ../test_vocab.json ../vocab/vocab-2016-09-10.json ../diff_between_test_and_model.txt --not_symmetric
``` 
This should return 23849 words that are in the test vocabularly but not in the model's. Next we want to see how many words are the model's vocabularly but not in the test's:
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab/vocab-2016-09-10.json ../test_vocab.json ../diff_between_model_and_test.txt --not_symmetric
``` 
This should return 625896 words, thus showing that there are a lot of I suppose redudant words.

## Difference between Training and Transformer ELMo model's vocabularly
Next to see how many of the model vocabularly words are not in the training vocabularly
``` bash
python vocab_comparison/comapre_vocabs.py ../training_vocab.json ../vocab/vocab-2016-09-10.json ../diff_between_train_and_model.txt --not_symmetric
``` 
This should return 1631869 words showing that there are a lot of words that the model does not have in it's vocabularly. Just to double check that there are not words in the model's vocabularly that do not exist in the training:
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab/vocab-2016-09-10.json ../training_vocab.json ../diff_between_model_and_train.txt --not_symmetric
``` 
This as expected return 3 words which should be `</S> <UNK> <S>` which is understandable. Thus this confirms that the models vocabularly does come from the training data but how were the model's words reduced from 2425337 unique words in the training data to 793468 the number of words in the models vocabularly (minus 3 from the special tokens mentioned at the start of this paragraph). We show that it was a frequency threshold where all words that occur at least 3 times goes into the models vocabularly and anything less is not included.

## How the Transformer ELMo model's vocabularly was reduced through frequency thresholding

First we need the following vocabularly coming from the following command:
``` bash
python vocab_comparison/comapre_vocabs.py ../training_vocab.json ../vocab/vocab-2016-09-10.json ../diff_between_train_and_model.json --not_symmetric --json
``` 
Where `../diff_between_train_and_model.json` will contain the vocabularly of all words that are not in the model vocabularly but is in the training.

Next we use the following script to find the highest occuring word count in the vocabularly from the training data:
``` bash
python vocab_comparison/frequency_words.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../diff_between_train_and_model.json --file_prefix news --highest
```
This should show the highest word and it's count as the following `Quatremer and 2`. Then to find the lowest word and count is from the model's vocabularly can be found using the following command:
``` bash
python vocab_comparison/frequency_words.py ../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ ../vocab/vocab-2016-09-10.json --file_prefix news
```
Which shows it as `Ayyubid and 3`, this proves that the theory behind the count threshold of the model's vocabularly.

This therefore draws to the end of how the Transformer ELMo models vocabularly was most likely created and that not all words in the vocabularly need to occur in the test or for that matter train set.

# To evaluate the Transformer ELMo using perplexity
As shown in the good tutorial about the model which can be found [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md) you can evaluate the model using the following command:
``` bash
allennlp evaluate --cuda-device 0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500] }}}' transformer-elmo-2019.01.10.tar.gz ../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```
NOTE: To switch it from using the GPU to CPU change the --cuda-device flag from 0 to -1.


However you may find that command to use too much RAM therefore you can use this slightly adapted command:
``` bash
allennlp evaluate --cuda-device 0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500], "max_instances_in_memory": 512, "batch_size": 128 }}}' transformer-elmo-2019.01.10.tar.gz ../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```
This basically changes the batch size used and the number of samples/instances that it stores in memory from 1024 and 32768 to 128 and 512 respectively.

# Difference in Vocabularly between the Pre-trained Transformer ELMo and the TDSA datasets
## Getting the vocabularly for each of the TDSA datasets
### Laptop
First we must generate the vocabularly from the TDSA datasets which can be done using the following commands:

``` bash
python vocab_comparison/tdsa_vocab.py laptop ../vocab_test_files/laptop_tdsa.json spacy tdsa_data/splits/
```
### Restaurant
``` bash
python vocab_comparison/tdsa_vocab.py restaurant ../vocab_test_files/restaurant_tdsa.json spacy tdsa_data/splits/
```
### Election
``` bash
python vocab_comparison/tdsa_vocab.py election ../vocab_test_files/election_tdsa.json spacy tdsa_data/splits/
```
## Words that are not in the Transformer ELMo vocabularly but in the TDSA datasets
Now we want to see how many of the model's vocabularly words are not in each of the TDSA words datasets:
### Laptop
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/laptop_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_laptop.txt --not_symmetric
```
224 (5.48%) different words out of 4085 unique words in the laptop dataset.
### Restaurant
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_restaurant.txt --not_symmetric
```
326 (6.37%) different words out of 5118 unique words in the restaurant dataset.
### Election
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/election_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_election.txt --not_symmetric
```
2012 (17.97%) different words out of 11194 unique words in the restaurant dataset.

As we can see the difference in unique words between the language model and the restaurant and laptop datasets is not that large in comparison to the election dataset. This is most likely due to the review domain which is more formal than the social media is more similar to how news data is written which is the domain of the 1 Billion word corpus.

## Words that are not in the Transformer ELMo vocabularly but are TARGET words in the TDSA datasets

Even though we have looked at the whole corpus vocabularlies of the TDSA datasets we are mainly interested in if target words are within the language models vocabularly as this is what we are changing within the data augmentation stage. Therefore now we are going to get only the target words from the TDSA datasets and compare the vocabularlys again to see the difference:

### Laptop

```bash
python vocab_comparison/tdsa_vocab.py laptop ../vocab_test_files/laptop_target_tdsa.json spacy tdsa_data/splits/ --targets_only
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/laptop_target_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_laptop_targets.txt --not_symmetric
python vocab_comparison/targets_affected.py laptop ../vocab_test_files/tdsa_diff_between_model_and_laptop_targets.txt spacy tdsa_data/splits/
python vocab_comparison/targets_affected.py laptop ../vocab_test_files/tdsa_diff_between_model_and_laptop_targets.txt spacy tdsa_data/splits/ --unique
```

Out of the 850 unique targets words (broken down into tokens therefore a multi word target will count as multi tokens here) 46 are not in the model's vocabularly and here is a sample of the target words the language model's vocabularly does not know:
``` python
["monitors", "intel", "iPhoto", "piece", "lcd", "GUI"]
```
As we can see this is pretty domain sepcific words and thus important words for the model to know hence the motivating reasons why we need to retrain the model on domain specific data. Out of these 46 target token words it affect 50 unique targets (out of 1295 unique target) of which these targets occur 61 times in total out of the whole dataset which is made up of 2951 samples. Thus in the grand scheme of things the model's vocabularly covers quiet well the targets in the laptop domain. Some examples of the targets affected:
``` python
["Sample of the targets affected", "WARRANTY COMPANY", "cusromer service center", "securitysoftware", "TYPING", "hardrive"]
```

### Restaurant

```bash
python vocab_comparison/tdsa_vocab.py restaurant ../vocab_test_files/restaurant_target_tdsa.json spacy tdsa_data/splits/ --targets_only
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/restaurant_target_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_restaurant_targets.txt --not_symmetric
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_model_and_restaurant_targets.txt spacy tdsa_data/splits/
python vocab_comparison/targets_affected.py restaurant ../vocab_test_files/tdsa_diff_between_model_and_restaurant_targets.txt spacy tdsa_data/splits/ --unique
```

Out of the 1027 unique targets words 74 are not in the model's vocabularly and here is a sample of the target words the language model's vocabularly does not know:
``` python
["atmoshere", "snapple", "Croquette", "Guacamole+shrimp", "falafal", "bruschettas"]
```
As we can see this is pretty domain sepcific words and thus important words for the model to know hence the motivating reasons why we need to retrain the model on domain specific data. Out of these 74 target token words it affect 73 unique targets (out of 1630 unique target) of which these targets occur 79 times in total out of the whole dataset which is made up of 4722 samples. Thus in the grand scheme of things the model's vocabularly covers quiet well the targets in the restaurant domain. Some examples of the targets affected:
``` python
["fresh mozzerella slices", "bruschettas", "measures of liquers", "omelletes", "lambchops"]
```

### Election

```bash
python vocab_comparison/tdsa_vocab.py election ../vocab_test_files/election_target_tdsa.json spacy tdsa_data/splits/ --targets_only
python vocab_comparison/comapre_vocabs.py ../vocab_test_files/election_target_tdsa.json ../vocab/vocab-2016-09-10.json ../vocab_test_files/tdsa_diff_between_model_and_election_targets.txt --not_symmetric
python vocab_comparison/targets_affected.py election ../vocab_test_files/tdsa_diff_between_model_and_election_targets.txt spacy tdsa_data/splits/
python vocab_comparison/targets_affected.py election ../vocab_test_files/tdsa_diff_between_model_and_election_targets.txt spacy tdsa_data/splits/ --unique
```

Out of the 1463 unique targets words 522 are not in the model's vocabularly and here is a sample of the target words the language model's vocabularly does not know:
``` python
["@steve_hawkes", "@BackBarwell", "dimelby", "ge15", "@candsalliance"]
```
As we can see this is pretty domain and medium (Twitter) sepcific words and thus important words for the model to know hence the motivating reasons why we need to retrain the model on domain and medium specific data. Out of these 522 target token words it affect 631 unique targets (out of 2190 unique target) of which these targets occur 1973 times in total out of the whole dataset which is made up of 11899 samples. Thus the model's vocabularly DOES NOT cover the targets in the election domain very well. Some examples of the targets affected:
``` python
["davidcameron", "ukip", "ukhousing", "@LeanneWood:'UKIP", "bbcqt"]
```
## Conclusion
In conclusion as we saw earlier with the tokens in each of the datasets in total the restaurant and laptop review datasets are covered quiet well in the language model for both the dataset words themseleves and more importantly the target words. However the Election Twitter dataset mainly because I believe of the Twitter part the language model does not cover well at all in the dataset nor the target words.