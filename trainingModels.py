import math # imporing math library to do logarithmic operations
# A unigram maximun likelihood model
def unigram_MLE (train_dictionary):
    train_dictionary_copy = dict(train_dictionary)
    train_dictionary_copy.pop("<s>")     #<s> symbol is not necessary to take account in unigram model
    total_tokens = sum(train_dictionary_copy.values())
    my_unigram = {}
    for key_word in train_dictionary_copy:
        my_unigram[key_word] = train_dictionary_copy[key_word] / total_tokens
    return my_unigram


# A bigram maximun likelihood model (and bigram smoothing, depending of "smoothing = True" otherwise only call to get bigrams )
def bigram_MLE (dictionary,input_file,smoothing):
    try:
        my_file = open(input_file, "r")
        lines = my_file.read().splitlines()
        dictionary_copy = dict(dictionary)
        bigram_dictionary = {}
        for line in lines:
            words_for_each_line = line.split()
            for i in range(len(words_for_each_line)-1):
                text = words_for_each_line[i]+(words_for_each_line[i + 1])
                if text in bigram_dictionary:
                    bigram_dictionary[text] += 1
                else:
                    bigram_dictionary[text] = 1
                text = ""
            words_for_each_line.clear()
        my_bigram = {}
        for line in lines:
            words_for_each_line = line.split()
            for i in range(len(words_for_each_line) - 1):
                keyword = str(words_for_each_line[i]) + str((words_for_each_line[i + 1]))
                if (smoothing):
                    my_bigram[keyword] = (bigram_dictionary[keyword] + 1) / (dictionary_copy[words_for_each_line[i]] + len (dictionary_copy))
                else:
                    my_bigram[keyword] = bigram_dictionary[keyword] / dictionary_copy[words_for_each_line[i]]
        my_file.close()
        return [my_bigram,bigram_dictionary] # Return the bigram (words and probabilities) and the bigram dictionary (words and bigram counts). Each word is as a key in the dictionary.
    except FileNotFoundError:
        print('File does not exist in bigram_MLE function')
        exit()

# Get the percentage of tokens and word types not seen in training data
def get_percentage_words(vocabulary_train, vocabulary_test):
    tot_not_seen_tokens = 0
    tot_not_seen_word_types = 0
    for word in vocabulary_test:
        if word not in vocabulary_train:
            tot_not_seen_tokens = tot_not_seen_tokens + vocabulary_test[word]
            tot_not_seen_word_types = tot_not_seen_word_types + 1

    percentage_not_seen_tokens = (tot_not_seen_tokens / sum(vocabulary_test.values())) * 100
    percentage_not_seen_word_types = (tot_not_seen_word_types / len(vocabulary_test)) * 100
    return [percentage_not_seen_tokens, percentage_not_seen_word_types]

# Computing log probabilities in unigram model
def compute_log_probabilities_unigram_MLE (sentence, vocabulary_train_unk):
    probability = {}
    log_propability_b2 = 0
    warning_notification = ""
    total_log_prob = 0
    vocabulary_train_unk_copy = dict(vocabulary_train_unk)
    my_sentence = (sentence.lower() + " </s>")
    result = ("Model: UNIGRAM" + '\n')
    result += ("Computing sentence: " + my_sentence + '\n')
    vocabulary_train_unk_copy.pop("<s>") # <s> symbol is not necessary to take account in unigram model
    total_tokens = sum(vocabulary_train_unk_copy.values())
    words_in_line = my_sentence.split()
    flag = 1
    M = len(words_in_line) # M is used for perplexity. (M is the number of words (tokens) in the test data)
    for word in words_in_line:
        if word in vocabulary_train_unk_copy:
            probability[word] = vocabulary_train_unk_copy[word] / total_tokens
            log_propability_b2 = math.log2(probability[word])  # Log probability (log base 2)
            result +=  ("Parameter to compute: P(" + str(word) + ")" + ", having c(" + str(word) + ") = " +
                   str(vocabulary_train_unk_copy[word]) + "/" + " c(total_tokens) = " +  str(total_tokens)
                   + ", the probability is: " + str(probability[word])
                   + " and the LOG probability is: " + str(log_propability_b2) + '\n')
        else: # In case the word in the sentence do not exist in the training data
            result += ("Parameter to compute: P(" + str(word) + ")" + " <UNK> TOKEN in the training data" + '\n')
            flag = 0
        if log_propability_b2 == 0:
            warning_notification += ( "See parameter " + "P(" + str(word) + ")" + '\n')
        else:
            total_log_prob += log_propability_b2
        log_propability_b2 = 0

    if flag: # if all the words have a probability
        l = total_log_prob / M # l is the average log
        perplexity =  2 ** (-l)
        result += ('\n')
        result += ("Total log probability: " + str(total_log_prob) + '\n')
        result += ("Average of total log probability: " + str(l) + '\n')
        result +=  ("Perplexity: " + str(perplexity) + '\n')
    else:
        result += ('\n' + "Calculations cannot be done due : " + warning_notification + '\n')
    result +=  ('\n')
    return result #return all text that will be printed in console

# Computing log probabilities in bigram model and bigram add 1 smoothing (if variable smoothing = True)
def compute_log_probabilities_bigram_MLE (sentence, my_bigram_vocabulary_train, vocabulary_train_unk,smoothing):
    warning_notification = '\n'
    my_sentence = ("<s> " + sentence.lower() + " </s>")
    result = ""
    if (smoothing):
        result += ("Model: BIGRAM ADD 1 SMOOTHING" + '\n')
    else:
        result +=  ("Model: BIGRAM" + '\n')
    result += ("Computing sentence: " + my_sentence + '\n')
    probability = {}
    total_log_prob = 0
    words_in_line = my_sentence.split()
    M = len(words_in_line) - 1  # M is used for perplexity
    flag = 1
    for i in range(len(words_in_line) - 1):
        keyword = words_in_line[i] + words_in_line [i+1] # for example: keyword = "<s>he"
        parameter_w1 = words_in_line[i + 1]
        parameter_w0 = words_in_line[i]
        given = words_in_line[i] # given variable represents the conditional word. for example: given = <s> in p (he | <s>)
        log_propability_b2 = 0
        if keyword in my_bigram_vocabulary_train:
            if given in vocabulary_train_unk:
                if (smoothing):
                    probability[keyword] = (my_bigram_vocabulary_train[keyword] + 1) / (vocabulary_train_unk[given] + len (vocabulary_train_unk))
                else:
                    probability[keyword] = my_bigram_vocabulary_train[keyword] / vocabulary_train_unk [given]

                log_propability_b2 = math.log2(probability[keyword])  # Log probability (log base 2)
                result += ("Parameter to compute: P(" + str(parameter_w1) + " | " + str(parameter_w0) + ")" + ", having c(" + str(parameter_w0) + "," + str(parameter_w1) + ") = " +
                      str(my_bigram_vocabulary_train[keyword]) + "/" + " c(" + str(parameter_w0) + "= " + str(vocabulary_train_unk [given])
                      + ", the probability is: " + str(probability[keyword])
                      + " and the LOG probability is: " + str(log_propability_b2) + '\n')
                total_log_prob += log_propability_b2
        else: # if keyword does not appear in my bigram vocabulary
            if (smoothing):
                probability[keyword] =  1 / (vocabulary_train_unk[given] + len (vocabulary_train_unk))
                log_propability_b2 = math.log2(probability[keyword])  # Log probability (log base 2)
                total_log_prob += log_propability_b2
                log_prob = log_propability_b2
            else: # Normal bigram
                flag = 0
                warning_notification += ("See parameter " + "P(" + str(parameter_w1) + " | " + str(parameter_w0) + ")" + " the log probability is UNDEFINED" + '\n')
                probability[keyword] = 0.0
                log_prob = "UNDEFINED"
            if given in vocabulary_train_unk:
                result += ("Parameter to compute: P(" + str(parameter_w1) + " | " + str(parameter_w0) + ")" + ", having c(" + str(
                    parameter_w0) + "," + str(parameter_w1) + ") = " +
                    str("<UNK>") + "/" + " c(" + str(parameter_w0) + ") = " + str(vocabulary_train_unk[given]) +
                    ", the probability is: " + str(probability[keyword])
                    + " and the LOG probability is: " + str(log_prob) + '\n')
            else:
                result += ("Parameter to compute: P(" + str(parameter_w1) + " | " + str(parameter_w0) + ")" + ", having c(" + str(
                    parameter_w0) + "," + str(parameter_w1) + ") = " +
                    str("<UNK>") + "/" + " c(" + str(parameter_w0) + ") = " + "UNK" +
                    ", the probability is: " + str(log_prob) + '\n')

        log_propability_b2 = 0

    if flag:  # if all the words have a probability
        l = total_log_prob / M  # l is the average log
        perplexity = 2 ** (-l)
        result += ('\n')
        result += ("Total log probability: " + str(total_log_prob) + '\n')
        result += ("Average of total log probability: " + str(l) + '\n')
        result += ("Perplexity: " + str(perplexity) + '\n')
    else:
        result += ('\n' +  "Calculations cannot be done due : " + warning_notification + '\n')
    result += ('\n')
    return result #return all text that will be printed in console

# Compute perplexity in unigram model
def compute_perplexity_unigram_MLE (my_unigram,input_file):
    my_file = open(input_file, "r")
    lines = my_file.read().splitlines()

    total_log_propability_b2 = 0
    M = 0  # M is used for perplexity. (M is the number of words (tokens) in the test data)
    for line in lines:
        words_for_each_line = line.split()
        for i in range(len(words_for_each_line)):
            if words_for_each_line[i] != '<s>': # it is not necessary to evaluate this symbol.
                total_log_propability_b2 += math.log2(my_unigram[words_for_each_line[i]])  # Log probability (log base 2)
        M += len(words_for_each_line)-1 # M is the number of tokens

    l = total_log_propability_b2 / M  # l is the average log
    perplexity = 2 ** (-l)
    return perplexity

# Compute perplexity in bigram model
def compute_perplexity_bigram_MLE (my_bigram,input_file):
    my_file = open(input_file, "r")
    lines = my_file.read().splitlines()
    flag = 1 # to know if there is not exist a bigram
    total_log_propability_b2 = 0
    M = 0  # M is used for perplexity. (M is the number of words (tokens) in the test data)
    for line in lines:
        words_for_each_line = line.split()
        for i in range(len(words_for_each_line) - 1):
            keyword = words_for_each_line[i] + words_for_each_line[i + 1]  # for example: keyword = "<s>he"
            if keyword in my_bigram:
                total_log_propability_b2 += math.log2(my_bigram[keyword])  # Log probability (log base 2)
            else:
                flag = 0 # we cannot calculate the log probability due there is not exist a bigram in the training data
        M += len(words_for_each_line)

    if flag == 0:
        perplexity = "THERE IS NOT POSSIBLE TO GET A PERPLEXITY DUE THERE ARE SOME BIGRAMS NOT FOUND IN TRAINING DATA"
    else:
        l = total_log_propability_b2 / M  # l is the average log
        perplexity = 2 ** (-l)
    return perplexity

# Compute perplexity in bigram smoothing model
def compute_perplexity_bigram_MLE_smoothing ( my_bigram_vocabulary_train, vocabulary_train_unk,input_file):
    my_file = open(input_file, "r")
    lines = my_file.read().splitlines()
    warning_notification = '\n'
    probability = {}
    total_log_prob = 0

    M = 0 # M is the number of words (tokens) in the test data
    for line in lines:
        words_in_line = line.split()
        for i in range(len(words_in_line) - 1):
            keyword = words_in_line[i] + words_in_line [i+1] # for example: keyword = "<s>he"
            parameter_w1 = words_in_line[i + 1]
            parameter_w0 = words_in_line[i]
            given = words_in_line[i] # given variable represents the conditional word. for example: given = <s> in p (he | <s>)
            log_propability_b2 = 0
            if keyword in my_bigram_vocabulary_train:
                probability[keyword] = (my_bigram_vocabulary_train[keyword] + 1) / (vocabulary_train_unk[given] + len (vocabulary_train_unk))
                log_propability_b2 = math.log2(probability[keyword])  # Log probability (log base 2)

                total_log_prob += log_propability_b2
            else: # if keyword does not appear in my bigram vocabulary
                    probability[keyword] =  1 / (vocabulary_train_unk[given] + len (vocabulary_train_unk))
                    log_propability_b2 = math.log2(probability[keyword])  # Log probability (log base 2)
                    total_log_prob += log_propability_b2

            log_propability_b2 = 0
        M += len(words_in_line)
        l = total_log_prob / M  # l is the average log
        perplexity = 2 ** (-l)
    return perplexity