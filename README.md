# Natural-Language-Processing - Language Modeling

Requirements

LANGUAGE MODELING
You will train several language models and will evaluate them on two test corpora. Three files are provided:
1. brown-train.txt
2. brown-test.txt
3. learner-test.txt

Each file is a collection of texts, one sentence per line. Brown-train.txt contains 26,000 sentences from the Brown corpus.
1 You will use this corpus to train the language models. The test corpora (brown-test.txt and learner-test.txt) will be used to evaluate the language models that you trained. brown-test.txt is a collection of sentences from the Brown corpus, different from the training data, and learner-test.txt are essays written by non-native writers of English that are part of the FCE corpus.

PRE-PROCESSING
Prior to training, complete the following pre-processing steps:
1. Pad each sentence in the training and test corpora with start and end symbols (you can use <s> and </s>, respectively).
2. Lowercase all words in the training and test corpora. Note that the data already has been tokenized (i.e. the punctuation has been split off words). 
3. Replace all words occurring in the training data once with the token <unk>. Every word in the test data not seen in training should be treated as <unk>.
  
TRAINING THE MODELS
Use brown-train.txt to train the following language models:
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

QUESTIONS

1. How many word types (unique words) are there in the training corpus? Include the padding symbols and the unknown token.
2. How many word tokens are there in the training corpus?
3. What percentage of word tokens and word types in each of the test corpora did not occur in training (before you mapped the unknown words to <unk> in training and test data)?
4. What percentage of bigrams (bigram types and bigram tokens) in each of the test corpora that did not occur in training (treat <unk> as a token that has been observed).
5. Compute the log probabilities of the following sentences under the three models (ignore capitalization and pad each sentence as described above). Please list all of the parameters required to compute the probabilities and show the complete calculation.
Which of the parameters have zero values under each model? Use log base 2 in your calculations. Map words not observed in the training corpus to the <unk> token.
• He was laughed off the screen .
• There was no compulsion behind them .
• I look forward to hearing your reply .
6. Compute the perplexities of each of the sentences above under each of the models.
7. Compute the perplexities of the entire test corpora, separately for the brown-test.txt and learner-test.txt under each of the models. Discuss the differences in the results you obtained.


RUNNING APPLICATION
The following lines are the instructions to run the application:

1.- Create a directory. For example "languageModeling"
2.- Put the following files in the same directory that was created in step 1:
    brown-train.txt
    brown-test.txt# File = Readme
    learner-test.txt
    main.py
    preProcessing.py
    trainingModels.py

3-. If it is used an IDE, set up the configuration to run the principal file called "main.py"
    Otherwise, run the following command in your terminal:
    python main.py

4.- The output will be displayed on the IDE console or terminal according to what was used.
    The project will generate the following files so as not to affect the original files.
    brown-train-preprocessed.txt
    brown-test-preprocessed.txt
    learner-test-preprocessed.txt
    answers.txt

5.- The answers are in answers.txt in the same directory.

Note: The project was interpreted by a python 3.7 version
