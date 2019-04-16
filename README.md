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

# Getting the data and models
## Target Dependent Sentiment Analysis (TDSA) data
### Getting the data and converting it into Train, Validation, and Test sets

Download the following datasets:

1. SemEval Restaurant and Laptop 2014 [1] [train](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and [test] and put the `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml` training files into the following directory `./tdsa_data` and do the same for the test files (`Laptops_Test_Gold.xml` and `Restaurants_Test_Gold.xml`)
2. Election dataset [2] from the following [link](https://ndownloader.figshare.com/articles/4479563/versions/1) and extract all of the data into the following folder `./tdsa_data/election`, the `election` folder should now contain the following files `annotations.tar.gz`, `test_id.txt`, `train_id.txt`, and `tweets.tar.gz`. Extract both the `annotations.tar.gz` and the `tweets.tar.gz` files.

Then run the following command to create the relevant and determinstic train, validaion, and test splits of which these will be stored in the following directory `./tdsa_data/splits`:

``` bash
python ./tdsa_data/generate_datasets.py ./tdsa_data ./tdsa_data/splits
```

This should create all of the splits that we will use throughout the normal experiments that are the baseline values for all of our augmentation experiments. This will also print out some statistics for each of the splits to ensure that they are relatively similar.

## 1 Billion word corpus Transformer ELMo language model
The [Transformer ELMo model](https://allennlp.org/elmo) that came from the following [paper](https://aclweb.org/anthology/D18-1179) was trained on the 1 Billion word corpus that can be downloaded from [here](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) and a good tutorial about the model can be found [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md). As the downloaded 1 billion corpus has already been tokenised and the Transformer ELMo model that has been downloaded comes with a fixed output vocabulary, we show that the vocabularly comes from only the training corpus and from words that have a frequency of at least 3 in the training corpus. We also find that not all the words in the test corpus are in the training corpus of the 1 billion word corpus.

NOTE: The model's vocabularly comes from un-tarring `transformer-elmo-2019.01.10.tar.gz` the model Transformer ELMo model download and going into the vocab folder.
NOTE: It is required to un-tar the `transformer-elmo-2019.01.10.tar.gz` as the vocabulary is required and the weights therefore we assume you have un-tared it into the following folder `../transformer_unpacked`

To see how we discovered how the Transformer ELMo model output vocabularly was discovered and how this vocabularly MORE IMPORTANTLY overlaps with the TDSA data and the TDSA target words look at the following [mark down file](./vocab_comparison/README.md)



## Yelp Data and filtering it

The [Yelp dataset](https://www.yelp.com/dataset) was accessed on the week starting the 1st of April 2019, this is important as Yelp releases a new dataset every year. We only used reviews that review businesses from the following categories `restaurants` `restaurant` `restaurants,` this was to ensure that the domain of the reviews were similar to the restaurant review domain (some reviews are about hospitals etc). To filter the reviews we ran the following command:
``` bash
python dataset_analysis/filter_businesses_by_category.py YELP_REVIEW_DIR/business.json business_filter_ids.json 'restaurants restaurant restaurants,'
```
Where `YELP_REVIEW_DIR` is the directory that contains the downloaded [Yelp dataset](https://www.yelp.com/dataset), `business_filter_ids.json` is the json file you want to store the filtered business ids that will only contain business ids that have come from the restaurant domain based on the 3rd argument `restaurants restaurant restaurants,` which specifies the categories the Yelp business must be within to be allowed in the `business_filter_ids.json` file.

### Converting the Yelp data into sentences and tokens.
Once we have the filtered data we are only interested in the text of the data as we want to fine tune the [Transformer ELMo model](https://allennlp.org/elmo) from the news corpus data it was original trained on to restaurant reviews. This is because restaurant reviews use a slightly different vocabularly (for examples see [this](./vocab_comparison/README.md)) and more than likely a different language style to news data. 

To convert the split Yelp reviews into split Yelp data that has been tokenised by spacy and only contains one sentence per line (same format as 1 Billion word corpus) run the following command:
``` bash
python dataset_analysis/to_sentences_tokens.py ../yelp/splits yelp
```
This will create three more files within the `../yelp/splits` directory; `split_train.txt`, `split_val.txt`, `split_test.txt`. This dataset will now be called **yelp sentences**

## Data Staistics of the language model datasets
### 1 Billion word corpus
As the original data the [Transformer ELMo model](https://allennlp.org/elmo) was trained on shuffled sentence from the one billion word corpus we wanted to see what the data statistics of this corpus was e.g. Max, Mean, Minimum sentence lengths etc therefore this can be seen by running the following command:
``` bash
python dataset_analysis/data_stats.py ../1-billion-word-language-modeling-benchmark-r13output/ billion_word_corpus --sentence_length_distribution ./images/sentence_distributions/1_billion_word_corpus.png
```
As we can seen the `max`, `mean`, and `min` token counts per sentence are 2818, 25.36, and 1 respectively. We can also see that very few sentences only contain one token (182) and further more very few contain only 2 tokens (19516). We can also see that the number of tokens that occur at least three times for the combined TRAIN and TEST set is 797,965 showing that the diversity in tokens is quiet large.

NOTE: `./images/sentence_distributions/1_billion_word_corpus.png` is just a graph of the sentence distribution.

### Yelp sentences

``` bash
python dataset_analysis/data_stats.py ../yelp/splits yelp_sentences --sentence_length_distribution ./images/sentence_distributions/yelp.png
```
As we can seen the `max`, `mean`, and `min` token counts per sentence are 937, 15.54, and 1 respectively. The fact that the sentences are shorter in general compared to the 1 billion word corpus could be the domain or the spacy sentence splitter. We can also see that the diversity of the tokens is smaller compared to the 1 billion word corpus as the number of tokens that occur at least three times in all of the data splits is only 260,150 compared to 797,965. Lastly we can also see that there are a lot more sentences with only 1 or 2 words in them of which I think this is most likely due to the sentence splitter and some what reviews in general e.g. review that just say good, great etc.

## Yelp sentence filtering based on sentence length

Based on these data statistics we are going to further filter the Yelp sentences dataset so that it only includes sentences that are at least 3 tokens long. We will also restrict the maximum sentence length to 500 due to GPU memory requirements. To do this run the following command:

``` bash
python dataset_analysis/filter_by_sentence_length.py ../yelp/splits yelp_sentences 3 500
```

This will create three more files within the `../yelp/splits` directory; `filtered_split_train.txt`, `filtered_split_val.txt`, and `filtered_split_test.txt`. These train, validation, and test splits are the final dataset that we will use to fine tune the Transformer ELMo model on for the restaurant review domain.


## How to run the Transformer ELMo model

command:


``` bash
allennlp evaluate --cuda-device -1 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500], "max_instances_in_memory": 512, "batch_size": 128 }}}' transformer-elmo-2019.01.10.tar.gz 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
```

## Fine tuning the Transformer ELMo model
In this section we show how you can fine tune the Transformer ELMo model to other domains and mediums.
### Yelp Restaurant Review dataset
Assuming that you have created `filtered_split_train.txt`, `filtered_split_val.txt`, and `filtered_split_test.txt` from the previous sections, we will use these datasets to fine tune the model. First we must create a new output vocabulary for the Transformer ELMo model to do this easily use the following command:
``` bash
python fine_tune_lm/create_lm_vocab.py fine_tune_lm/training_configs/yelp_lm_vocab_create_config.json ../yelp_lm_vocab
```
Where `../yelp_lm_vocab` is a new directory that stores only the vocabulary files, of which the vocabulary that will be used can be found here `../yelp_lm_vocab/tokens.txt`.

To make the training process quicker it is advised that you split the training corpus up into serveal files this can be done using the following command:
``` bash
python fine_tune_lm/split_dataset.py ../yelp/splits/filtered_split_train.txt ../yelp/splits/filtered_train_dir/ 40
```
Where `../yelp/splits/filtered_split_train.txt` is the file that stores all of the training data and `../yelp/splits/filtered_train_dir/` is the new directory that will store all of the training data but over 40 files. 40 is just an arbitary number any number can be chosen.

To train the model run the following command (This will take a long time):
```
allennlp train fine_tune_lm/training_configs/yelp_lm_config.json -s ../yelp_language_model
```
Where `../yelp_language_model` is the directory that will save the language model to.
