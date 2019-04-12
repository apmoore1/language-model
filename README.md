When we are splitting the Yelp dataset we are going to keep 8% of the reviews for test and validation sets and 84% for training which is very similar number of articles (8.3%) that were used for the [wikiText2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)


from what I gather in the hugginface version they add <eos> at the start as the starting symbol and in the original version they do not but they both share that <eos> happens at the end of a line even if the end of the line happens multiple times thus this can occur <eos> <eos> this is only for the wt103 dataset also I think the wt103 dataset is the only one where the model only knows the words in the training data and not in any of the other datasets.


From the paper the transformer XL states that there are two problems:
1. With the traditional transformer you have to have a fixed length context window due to memory and compute reasons, thus with this you ignore the extended context from the other related contexts which is the context fragmentation problem
2. Having this fixed context and segements will not allow you to learn long term dependency between words that are across segements, this is probably more important in story books/novels.


We are going to use the pre-trained model from wikiText-103 as it has long term dependencies, 103M training tokens, 28K articles, average length of 3.6K per article. 



I was wondering what the license terms were for the Yelp dataset and then also the amazon dataset and then I was thinking for Twitter as well.

The Yelp data license I think this is the key part in section 3:
"Term to use, access, and create derivative works of the Data in electronic form for academic purposes only." I think this could be an interesting problem as a language model I suppose could generate an identical review but without knowing: "i.e. you may not publicly display any of the Data to any
third party, especially reviews and other user generated content, as this is a private data set
challenge and not a license to compete with or disparage with Yelp" I also think this part of the aggrement could be legally tricky with respect to publishing the model "rent, lease, sell, transfer, assign, or sublicense, any part of the Data" Section 5 also falls into this with respect to who would own the model as in the does section 5 just say that the only company that could have any rights over it is the Yelp company.


So from what I have gathered the reason why the test data in Wiki103 is <eos> is because that is the symbol that is applied for new line and the first token in the test data is a new line.


I think the next step would be to just split the yelp dataset into train, val and test based on previous work in the area on the size of train, val, and test.

The next step after that would be to look at the size of the dataset by the token size where we will split based on whitespace. Then I think we will look into which model to use as the TransformerXL from what I have just gathered only uses token level information and does not encode any characters which is problematic.

# 1 Billion word corpus and Testing the original Transformer ELMo language model
The [Transformer ELMo model](https://allennlp.org/elmo) that came from the following [paper](https://aclweb.org/anthology/D18-1179) was trained on the 1 Billion word corpus that can be downloaded from [here](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) and a good tutorial about the model can be found [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md). As the downloaded 1 billion corpus has already been tokenised and the Transformer ELMo model that has been downloaded comes with a fixed output vocabulary, we show that the vocabularly comes from only the training corpus and for words that have a frequency of at least 3. We also find that not all the words in the test corpus are in the training corpus of the 1 billion word corpus. Below are some commands to show these findings:

NOTE: The model's vocabularly comes from un-tarring `transformer-elmo-2019.01.10.tar.gz` the model Transformer ELMo model download and going into the vocab folder.

#### Difference between the 1 Billion test and training vocabularlies
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

#### Difference between the Transformer ELMo model's vocabularly and the 1 Billion training and testing Vocabularlies.
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


Next to see how many of the model vocabularly words are not in the training vocabularly
``` bash
python vocab_comparison/comapre_vocabs.py ../training_vocab.json ../vocab/vocab-2016-09-10.json ../diff_between_train_and_model.txt --not_symmetric
``` 
This should return 1631869 words showing that there are a lot of words that the model does not have in it's vocabularly. Just to double check that there are not words in the model's vocabularly that do not exist in the training:
``` bash
python vocab_comparison/comapre_vocabs.py ../vocab/vocab-2016-09-10.json ../training_vocab.json ../diff_between_model_and_train.txt --not_symmetric
``` 
This as expected return 3 words which should be `</S> <UNK> <S>` which is understandable. Thus this confirms that the models vocabularly does not come from the training data but how were the model's words reduced from 2425337 unique words in the training data to 793468 the number of words in the models vocabularly (minus 3 from the special tokens mentioned at the start of this paragraph). We show that it was a frequency threshold where all words that occur at least 3 times goes into the models vocabularly and anything less is not included.

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

#### To evaluate the Transformer ELMo using perplexity
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

## Yelp Data

The [Yelp dataset](https://www.yelp.com/dataset) was accessed on the week starting the 1st of April 2019, this is important as Yelp releases a new dataset every year. We only used reviews that review businesses from the following categories `restaurants` `restaurant` `restaurants,` this was to ensure that the domain of the reviews were similar to the restaurant review domain (some reviews are about hospitals etc). To filter the reviews we ran the following command:
``` bash
python YELP_REVIEW_DIR/business.json business_filter_ids.json 'restaurants restaurant restaurants,'
```
Where `YELP_REVIEW_DIR` is the directory that contains the downloaded [Yelp dataset](https://www.yelp.com/dataset), `business_filter_ids.json` is the json file you want to store the filtered business ids that will only contain business ids that have come from the restaurant domain based on the 3rd argument `restaurants restaurant restaurants,` which specifies the categories the Yelp business must be within to be allowed in the `business_filter_ids.json` file.

## How to run the Transformer ELMo model

command:


``` bash
allennlp evaluate --cuda-device -1 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500], "max_instances_in_memory": 512, "batch_size": 128 }}}' transformer-elmo-2019.01.10.tar.gz 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```

## What is the vocab comparison folder
The vocab comparison folder contains script that will create a `vocab.json` file (or whatever you would like to call it) that will contain a list of all vocabulary terms given a corpus to create the vocabularly from and a tokeniser to `create/split` the words within the corpus.

To create a vocabulary file from the 1 Billion word held out corpus:
``` bash
python create_vocab.py 1-Billion-Held-Out-Corpus-File-Path Vocab-File-Path whitespace
python vocab_comparison/create_vocab.py ../1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 ../vocab_1_billion_held_out whitespace

```
Given the location of the 1 Billion word held out corpus it will create the json vocabulary file at the given path using the whitespace tokeniser.
