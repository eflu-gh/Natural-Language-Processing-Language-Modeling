# 1.- Pad each sentence in the training and test corpora with start <s> and end symbols </s>
# 2- Lowercase all words in the training and test corpora.
def pads_and_lower_symbols (input_file, output_file):
    try:
        text = ""
        my_file = open(input_file, "r")
        lines = my_file.read().splitlines()
        for line in lines:
            text += ("<s> " + line.lower() + " </s>" + '\n')
        my_file = open(output_file, "w")
        my_file.writelines(text)
        my_file.close()
    except FileNotFoundError:
        print('File does not exist in pads_and_lower_symbols function')
        exit()

# Get the number of times where a word appears in a given file
def getFrequenciesPerWord (file_name):
    dictionary = {}
    my_file = open(file_name, "r")
    words = my_file.read().split()
    for keyword in words:
        if keyword in dictionary:
            dictionary[keyword] += 1 # There is already a keyword
        else:
            dictionary[keyword] = 1 # keyword is new
    my_file.close()
    return dictionary


# Replace all words occurring in the training data once with the token <unk>.
def replace_words_training_data (filename,dictionary):
    try:
        my_file = open(filename, "r")
        lines = my_file.read().splitlines()
        my_file = open(filename, "w")
        # I replace the <unk> for each line in order to conserve the original format (a sentence for each line)
        for line in lines:
            words_for_each_line = line.split()
            for i in range(len(words_for_each_line) - 1):
                if dictionary[words_for_each_line[i]] == 1:
                    words_for_each_line[i] = "<unk>"
            my_file.write(' '.join(words_for_each_line) + '\n')
            words_for_each_line.clear()
        my_file.close()
    except FileNotFoundError:
        print('File does not exist in replace_words_training_data function')
        exit()


# Every word in the test data not seen in training should be treated as <unk>.
def replace_words_test_data (filename, train_dictionary):
    try:
        my_file = open(filename, "r")
        lines = my_file.read().splitlines()
        my_file = open(filename, "w")
        for line in lines:
            words_for_each_line = line.split()
            for i in range(len(words_for_each_line)):
                if words_for_each_line[i] not in train_dictionary:
                    words_for_each_line[i] = "<unk>"
            my_file.write(' '.join(words_for_each_line) + '\n')
            words_for_each_line.clear()
        my_file.close()
    except FileNotFoundError:
        print('File does not exist in replace_words_test_data function')
        exit()

# Preparing the data unseen for question 5 (some sentences)
def mapping_words_unseen (list_sentences, frequencies_train_unk):
    new_list = []
    new_sentence = ""
    for sentence in list_sentences:
        sentence = (sentence.lower())
        words = sentence.split()
        for i in range(len(words)):
            if words[i] not in frequencies_train_unk:
                words [i] = "<unk>"
            new_sentence = ' '.join(words)
        new_list.append(new_sentence)
    return new_list