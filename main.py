from preProcessing import *
from trainingModels import *

input_file = "brown-train.txt"
output_file = "brown-train-preprocessed.txt"
pads_and_lower_symbols(input_file,output_file)
# vocabulary is a set of words W1, W2...Wn such that contains each word and its frequency in the training file (brown-train.txt)
# vocabulary includes <s> and </s> symbols without <unk>
frequencies_train_nounk = getFrequenciesPerWord (output_file)

#Process the brown-train file, adding <unk> where a word is shown only once
replace_words_training_data(output_file,frequencies_train_nounk)
frequencies_train_unk = getFrequenciesPerWord (output_file)

##################################################################################################################################
input_file = "brown-test.txt"
output_file = "brown-test-preprocessed.txt"
pads_and_lower_symbols(input_file,output_file)
frequencies_brown_test_nounk = getFrequenciesPerWord (output_file)

replace_words_test_data (output_file,frequencies_train_unk)

frequencies_brown_test_unk = getFrequenciesPerWord (output_file)
brown_test_unk= bigram_MLE (frequencies_brown_test_unk,output_file,False)

##################################################################################################################################

input_file = "learner-test.txt"
output_file = "learner-test-preprocessed.txt"
pads_and_lower_symbols(input_file,output_file)
frequencies_learner_test_nounk = getFrequenciesPerWord (output_file)

replace_words_test_data (output_file,frequencies_train_unk)

frequencies_learner_test_unk = getFrequenciesPerWord (output_file)
learner_test_unk= bigram_MLE (frequencies_learner_test_unk,output_file,False)

##################################################################################################################################
#my n-gram is a dictionary sctructure where I have a word and its probability.
output_file = "brown-train-preprocessed.txt"
my_unigram = unigram_MLE (frequencies_train_unk)
my_bigram_train = bigram_MLE (frequencies_train_unk,output_file,False)
my_bigram_train_smoothing = bigram_MLE (frequencies_train_unk,output_file,True)
##################################################################################################################################

answers = open("answers.txt","w")
result = ""

#Question 1
file = "brown-train-preprocessed.txt"
word_types_pad_and_unk = getFrequenciesPerWord (file)
result += ("Answer 1: Number of Word types (unique words) in training corpus including the padding symbols and the unknown token " + str(len(word_types_pad_and_unk)) + '\n' + '\n')


#Question 2
result += ("Answer 2: Number of WORD TOKENS in the training corpus: " + str(sum(word_types_pad_and_unk.values())) + '\n' + '\n')

#Question 3
percentage_words_brown_test = get_percentage_words(frequencies_train_nounk,frequencies_brown_test_nounk)
percentage_words_learner_test = get_percentage_words(frequencies_train_nounk,frequencies_learner_test_nounk)

result +=  ("Answer 3.1: The percentage of word tokens in brown test and not in training data is: " + str(round(percentage_words_brown_test[0],4)) + '\n')
result +=  ("Answer 3.2: The percentage of word types in brown test and not in training data is: " + str(round(percentage_words_brown_test[1],4)) + '\n')
result +=  ("Answer 3.3: The percentage of word tokens in learner test and not in training data is: " + str(round(percentage_words_learner_test[0],4)) + '\n')
result +=  ("Answer 3.4: The percentage of word types in learner test and not in training data is: " + str(round(percentage_words_learner_test[1],4)) + '\n' + '\n')

#Question 4
percentage_words_brown_test = get_percentage_words(my_bigram_train[1],brown_test_unk[1])
percentage_words_learner_test = get_percentage_words(my_bigram_train[1],learner_test_unk[1])

result +=  ("Answer 4.1: The percentage of bigram tokens in Brown Test and not in training data is: " + str(round(percentage_words_brown_test[0],4)) + '\n')
result +=  ("Answer 4.2: The percentage of bigram types in Brown Test and not in training data is: " + str(round(percentage_words_brown_test[1],4)) + '\n')
result +=  ("Answer 4.3: The percentage of bigram tokens in Learner Test and not in training data is: " + str(round(percentage_words_learner_test[0],4)) + '\n')
result +=  ("Answer 4.4: The percentage of bigram types in Learner Test and not in training data is: " + str(round(percentage_words_learner_test[1],4)) + '\n' + '\n')

#Question 5 and 6 (perplexity can be calculated from the log probability)
result +=  ("Answers 5 and 6:" + '\n')
list_sentences = ["He was laughed off the screen ."]
list_sentences.append("There was no compulsion behind them .")
list_sentences.append("I look forward to hearing your reply .")

list_sentences = mapping_words_unseen (list_sentences, frequencies_train_unk)

#Computing log probabilities for each model already computed (unigram, bigram and bigram with add-one smoothing
for sentence in list_sentences:
    result += compute_log_probabilities_unigram_MLE (sentence,frequencies_train_unk) + '\n'
    result += compute_log_probabilities_bigram_MLE (sentence,my_bigram_train[1],frequencies_train_unk,False) + '\n' # without smoothing
    result += compute_log_probabilities_bigram_MLE(sentence, my_bigram_train[1], frequencies_train_unk,True) + '\n' # add 1 smoothing

#Question 7
result +=  ("Answers 7" + '\n')
output_file = "brown-test-preprocessed.txt"
result +=  ("Test Corpora: BROWN-TEST" + '\n')
result += ('Perplexity under UNIGRAM model: ' +
      str(compute_perplexity_unigram_MLE (my_unigram,output_file)) + '\n')
result += ('Perplexity under BIGRAM model: ' +
      str(compute_perplexity_bigram_MLE (my_bigram_train[0],output_file)) + '\n')
result += ('Perplexity under BIGRAM 1-ADD SMOOTHING model: ' +
      str(compute_perplexity_bigram_MLE_smoothing (my_bigram_train[1], frequencies_train_unk,output_file)) + '\n')

result +=  ('\n' + '\n')

output_file = "learner-test-preprocessed.txt"
result +=  ("Test Corpora: LEARNER-TEST" + '\n')
result += ('Perplexity under UNIGRAM model: '
      + str(compute_perplexity_unigram_MLE (my_unigram,output_file)) + '\n')
result += ('Perplexity under BIGRAM model: '
      + str(compute_perplexity_bigram_MLE (my_bigram_train[0],output_file)) + '\n')
result += ('Perplexity under BIGRAM 1-ADD SMOOTHING model: '
      + str(compute_perplexity_bigram_MLE_smoothing (my_bigram_train[1], frequencies_train_unk,output_file)) + '\n')

print (result)
answers.write (result)