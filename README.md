# Natural Language Processing Mini-Challenge 1 

## Setup

* Clone the repository:

```bash
git clone https://github.com/BrunoKreiner/nlp-mc-1
cd nlp-mc-1
```

* Create and activate a new conda environment and install the requirements:

```bash
conda create --name yourprojectenv python=3.9
conda activate yourprojectenv
pip install -r requirements.txt
```

## Objective

As part of my 5th-semester project, 49985 songs have been classified with a list of genres pertaining to the artist of the particular song. For each song and its artist(s), there also exists some metadata. The objective is to classify the correct genre(s) of each song based on their lyrics since crawling the lyrics of those songs was a big task in the project. The metadata will not be included in the classification in the first step but could be further investigated. 

## Preprocessing

The lyrics have already been cleaned from non alphanumeric characters. The cleanup can be seen in the following figure: 
<img src="./reports/cleanup-flowchart.jpg" width=50% height=50%>

The preprocessing in [preprocess.ipynb](./notebooks/preprocess.ipynb) is aimed towards simplifying genre classification by reducing the total number of classes to N = 10 by using the most common word in a list of genres.

### Genre List Preprocessing

For every song there exists a list of genre that actually pertains to the artist of that genre. If there are more than one artists, only the genre of the first artist is included. To simplify classification, we focus on the most common word in the genre list which in the example ['emo rock', 'progressive rock', 'metal'] would be 'rock'. The genre of that 'song' (actually the artist) would then be 'rock'. Spotify doesn't provide genre specific to one song, so we have to go from there. Some genre lists only have one element in the form of ['pop rock']. In that case the last word will be chosen. Some of the genre lists have ambiguous most common words. For example ['esoteric house', 'esoteric pop', 'alternative pop']. Here 'esoteric' will be chosen, since it's the most common word. The counter might choose 'esoteric' over 'pop' since it's the first occurence in the list or maybe because it's ordered by alphabet. The counter is taken from ollections.Counter. The 10 most common genres will then be used to build the train / test split. The same can be done on the 5, 25 or 50 most common genres, but for simplicity we focus on the 10 most common genres.

## Using TF-IDF and linear SVC

Term Frequency Inverse Document Frequency is a technique to quantify the importance of a term in a document. In our example, the documents talked about in TF-IDF theory, are the song lyrics. TF-IDF gives more weight to terms that are important but occur rarely in the corpus as a whole. Terms that occur frequently in one document but also occur frequently in all of the documents are seen as less important.

First, the term frequency of each term is calculated for each document. Then, the document frequency is calculated which is how many times the term is seen over all documents. The logarithm is applied as such: log ( number of documents / document-frequency ) to give rare terms a higher score. Then, for each document, the final TF-IDF is calculated using the previously calculated log of inverse document frequency multiplied by the term frequency.

This matrix is given to the LinearSVC as input. It produces a linear support vector machine model that fits the data.
LinearSVC finds a hyperplane that separates the data into specific classes (in our case song genres). 

In [simpler_models.ipynb](./notebooks/simpler_models.ipynb), I used a Pipeline using sci-kit learn's FeatureUnion to calculate different TF-IDF vectorizers. I understood this as calculating multiple matrices with different TF-IDF hyperparameters and feeding all of that into a LinearSVC(). This effectively augments the data. The hyperparameters are the n-gram range. This means the matrices consider consecutive words as one single term. How many consecutive words are considered is the "n" in n-gram. For example, "ngram_range=(1, 2)" would consider both unigrams and bigrams.

The model doesn't show a very good result which will be discussed in ## Results.

## BERT + Pytorch

In a second step, I stepped away from TF-IDF and used a pre-trained BERT tokenizer in [pytorch_transformers.ipynb](./notebooks/pytorch_transformers.ipynb), and put the lyrics through a BERT model. Specifically, the sentence-transformer library was used to embed lyrics in one step. This means the complete song text is embedded to one embedding instead of using tokenwise embedding. The output is an encoded vector of the lyrics. The output is then passed to a neural network for classification. Multiple Neural Networks and BERT models were tested and accuracies + f1-scores are discussed in *Results*. These BERT models were not trained further and only the added layers were trained. The BERT models only act as "backbones". These experiments were done in [pytorch_transformers.ipynb](./notebooks/pytorch_transformers.ipynb). There, different BERT models were loaded and tested using different neural nets. All models were trained for 25 epochs with a learning rate of 0.001 and a batchsize of 512. The loss is the cross entropy loss with balancing rescaling weights since the classes are imbalanced (rock music occurs much more than christian music).

BERT (Bidirectional Encoder Representations from Transformers) is a neural network architecture that was introduced in 2018 by Google researchers. It is a pre-trained transformer-based model that is trained on large amounts of text data to create embeddings, which can then be fine-tuned on specific tasks, such as text classification or question-answering. The core of BERT is a transformer, which is a type of neural network that processes sequential data, such as text. Unlike traditional sequential models, transformers are designed to allow for parallel processing of the input sequence, making them much faster and more efficient. Transformers work by breaking the input sequence into smaller chunks called "tokens", and then processing each token in parallel. In BERT the words are also tokenized based on their frequencies. In the BERT models used in this project, one was pre-trained using Contrastive Learning on song lyrics data and named lyrics-bert. It is very small compared to other models. Therefore, the baseline distilbert-base-uncased shows a bit better results. 

## Results

Plotting all pytorch model's training history (BERT based), we can see a couple of things: 

![Model Training History](./reports/model_history.png) 

We can see that all the models fail to reach a test accuracy of over 45%. Since we have 10 classes, the minimum accuracy achievable by chance is 1/10, so 45% is okay for the first step. This score is the same for the f1-score. The best model is the MLP with batch normalization that uses distilbert-base-uncased as the backbone. The models with lyricsbert as the backbone were slightly worse but that's also because this model is 3 times smaller than distilbert-base-uncased. The test_accuracy plot next to the train_accuracy one shows that the train accuracy is much higher in most models meaning that overfitting is prevalent. Which embedding model used is not that important, because the accuracy stays low. We can only see that the convolution networks weren't that good and models with fully connected layers were overfitting pretty hard. Maybe different batch sizes and learning rates would have changed the outcome, but we can also assume that the accuracies wouldn't have improved by a lot. The test f1 score behaves almost identically to the test accuracy. For the train and test losses at the bottom, we can see that the models converge very quickly. We can see that the models that overfit the training accuracy, actually converge the fastest in the test loss. Interesting to note is that their test loss goes up again. 

For the simpler models using Tf-IDF and Support Vector Machines, the accuracies and f1-scores look very similar as shown in the confusion matrices in [simpler_models.ipynb](./notebooks/simpler_models.ipynb). We can assume that changing the way we process the texts into tokens doesn't improve the models too much. If we look at the confusion matrices, the way to go seems to be augmenting the data, getting more data, maybe few-shot learning, and tackling the problem of class imbalance more at the data level. The models that used the library nlpaug for data augmentation didn't show better results, but they could be trained for a longer time, to make data augmentation more worth it. To improve classification, more data can be crawled or models suited for class imbalance can be used.

In the confusion matrices in the scikit-learn models ([simpler_models.ipynb](./notebooks/simpler_models.ipynb)) and in the pytorch models ([pytorch_transformers.ipynb](./notebooks/pytorch_transformers.ipynb)), we can see that the SVC models show very similar results to all top performing pytorch models. They all predict rock and pop (and maybe hip-hop) the most even if it's not correct. The better ones make more evenly distributed bad predictions. Rock and pop are the two most commonly occurring classes so it makes sense that bad predictions fall into either rock or pop.

### Manual Evaluation

In [manual_evaluation.ipynb](./notebooks/manual_evaluation.ipynb), we can look at a couple of specific things in detail. In this notebook, the "fc-with-batch-normalization-bert-base" model was used and analyzed. Looking at the classification report, we can see that the model performs best on the "metal" genre with a precision of 0.61 and a recall of 0.57. This could be due to the easily distinguishable language of the genre. On the other hand, the model struggles with "punk", "rap", "indie" and "soul". These genres might have more overlap with other genres in terms of song text. To improve performance on these genres, we could consider using genre-specific models, maybe trained in a binary setting (is this punk or not?) or similar. 

The accuracy is also imbalanced across genres. This is due to the class imbalance in the dataset as well as the hard-to-distinguish lyrics. We could use oversampling on some genres like "christian" music which has only 265 occurrences vs. 1113 rock songs. Nevertheless, the f1-score of "christian" music is surprisingly high though in comparison to other genres.

#### Relating it back to the data

The lyrics themselves look quite complex and can be interpreted in many ways. For example, the lyrics that were misclassified as "rock" but were actually "indie" contain emotional language and metaphors, which could be common in both genres. The model might be confused by the use of certain words or phrases that are common in both genres. The misclassified example that was predicted as "soul" but is actually "pop", contains a strong emotional component which might be a common characteristic of both "pop" and "soul" music. It would probably be pretty hard for a human to classify the lyrics as well. It might be hard to push a model to its full potential thinking about these things and more complex modeling techniques might be necessary.

To improve the model's understanding, sentiment analysis or semantic analysis could be used. We could also use the metadata provided by the data. The overlap between genres is also apparent in terms of lyrical content. For example, "soul" and "pop" often share similar themes of love and heartbreak.

The length of the lyrics could also be a factor in the model's predictions. Longer lyrics provide more context for the model to understand the song's theme but the model needs to be "powerful" enough to understand such a long input. This depends on the sentence-transformer library and its inner workings. The max input sequence length could also make a difference. 

#### Conclusion

In conclusion, the f1-score is pretty low which also means there is significant room for improvement. By refining the way data is treated for training and predictions at the data level, the models could be improved. The preprocessing step simplifies the genre classification by focusing on the most common word in the genre list. This approach has its limitations as well. For example, "rock" and "pop rock" are quite different, but both would be classified as "rock" with the preprocessing used. This could lead to misclassification. Rock is one of the genres that is misclassified as other genres the most across all different genres. 

## Extras

- In [text_processing.ipynb](./notebooks/text_processing.ipynb), the classification performance was tested when deleting contractions, using lemmatization (reducing words to their stem word ("running" to "run")), deleting stopwords, and combining all of these techniques. No improvements could be observed.

- When looking at fine-tuning a BERT model without freezing its layer as we did earlier, the model didn't train. The code for this is in [full_fine_tune_lyrics_bert.ipynb](./notebooks/full_fine_tune_lyrics_bert.ipynb). Adjusting the parameters there might yield better results in the end. 
