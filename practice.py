import joblib
import os
import sys
import pickle
from nltk.tokenize import WhitespaceTokenizer
from datasets import load_dataset
import sklearn_crfsuite
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import sentencepiece as spm

class BaselineCRF:
    """
    Baseline CRF implemented using sklearncrf_suite
    """
    def __init__(self, language: str):
        """
        The constructor function for the baseline CRF that takes in the name of the language and looks for the file
         corresponding to that language in the morphology folder
        :param language: string of the language that particular model should focus on
        """
        self.input_files = ["./data/zu.clean.train.conllu",
                            "./data/zu.clean.dev.conllu",
                            "./data/zu.clean.test.conllu"]
        self.language = language
    
    def to_use_surface_crf(self):
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                result = []
                morph = ''
                tag = False

                for char in content[1]:
                    # Surface Segmentation
                    if char == '-':
                        result.append(morph)
                        morph = ''
                    else:
                        morph += char

                if morph.strip():
                    result.append(morph.strip())

                label = ''
                for morph in result:
                    if len(morph) == 1:
                        label += 'S'
                    else:
                        label += 'B'
                        for i in range(len(morph) - 2):
                            label += 'M'
                        label += 'E'

                dictionaries[counter][content[0]] = label

            counter += 1

        best_epsilon, best_max_iteration = 0, 0
        maxF1 = 0
        for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160]:
                X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
                X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
                crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                crf.fit(X_training, Y_training)

                Y_predict = crf.predict(X_dev)
                Y_dev = MultiLabelBinarizer().fit_transform(Y_dev)
                Y_predict = MultiLabelBinarizer().fit_transform(Y_predict)
                f1 = f1_score(Y_dev, Y_predict, average='micro')
                if f1 > maxF1:
                    print(f1)
                    maxF1 = f1
                    best_epsilon = epsilon
                    best_max_iteration = max_iterations

        print(best_max_iteration)
        print(best_epsilon)
        print(maxF1)
        

        X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
        return crf
    
def surface_segment_data_preparation(word_dictionary: {str, str}):
    """
    This Method is used to generate features for the crf that is performing the surface segmentation
    :param word_dictionary: A word dictionary with the keys being the words and the value being the list of labels
    corresponding to each character
    :return: List of features, List of Correct Labels, The word as a list
    """
    X = []
    Y = []
    words = []
    for word in word_dictionary:
        word_list = []
        word_label_list = []
        for i in range(len(word)):
            gram_dict = {}
            gram_arr = []

            ### Unigram
            # gram_dict[word[i]] = 1
            gram_dict["uni_" + word[i]] = 1
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 3:
                        gram_dict["tri_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 5:
                        gram_dict["five_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            if word[i] in 'aeiou':
                gram_dict["vowel"] = 1
            else:
                gram_dict["const"] = 1

            if word[i].isupper():
                gram_dict["upper"] = 1
            else:
                gram_dict["lower"] = 1

            word_list.append(gram_dict)
            try:
                word_label_list.append(word_dictionary[word][i])
            except:
                print(word_label_list)
                print(word_dictionary[word])
                print(word)
        X.append(word_list)
        Y.append(word_label_list)
        words.append([char for char in word])
    return X, Y, words


def surface_segment_data_active_preparation(word_list: [str]):
    X = []
    for word in word_list:
        word_list = []
        for i in range(len(word)):
            gram_dict = {}
            gram_arr = []

            ### Unigram
            # gram_dict[word[i]] = 1
            gram_dict["uni_" + word[i]] = 1
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 3:
                        gram_dict["tri_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 5:
                        gram_dict["five_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            if word[i] in 'aeiou':
                gram_dict["vowel"] = 1
            else:
                gram_dict["const"] = 1

            if word[i].isupper():
                gram_dict["upper"] = 1
            else:
                gram_dict["lower"] = 1

            word_list.append(gram_dict)

        X.append(word_list)
    return X

def eval_morph_segments(predicted, target):
    """
    Method used to calculate precision, recall and f1 score 
    may be of different lengths
    :param predicted: the list of predicted labels
    :param target: the list of actual values
    :return: precision, recall and f1 score
    """
    print(predicted[0])
    print(target[0])
    pred_res = []
    targ_res = []
    #finding morpheme predicted and actual morpheme boundaries
    for (pred, targ) in zip(predicted, target):
        for index, (p, m) in enumerate(zip(pred, targ)):
            if index>0 and index< (len(pred)-1) and p=="E":
                pred_res.append(1)
            else:
                pred_res.append(0)
            if index>0 and index<(len(targ)-1) and m=="E":
                targ_res.append(1)
            else:
                targ_res.append(0)
    print(pred_res[0:len(predicted[0])])
    print(targ_res[0:len(predicted[0])])
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for p, t in zip(pred_res, targ_res):
        if t == p == 1:
            true_positives+=1
        elif p==1 and t==0:
            false_positives+=1
        elif p==0 and t==1:
            false_negatives+=1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f_score
zulu_surface= joblib.load('./models/isiZuluSurfaceModel.sav')
tokenizer = WhitespaceTokenizer()
batch = tokenizer.tokenize("Ziningi izinhlobo ezehlukene zokuphathwa kwephrojekthi ezisetshenziswa emkhakheni wezentuthuko.")
features = surface_segment_data_active_preparation(batch)
labels = zulu_surface.predict(features)
tmp = []
for word, label in zip(batch, labels):
      for i in range(len(label)):
          if label[i] == "S" or label[i] == "E":
              tmp.append(word[i])
              tmp.append(" Ġ")
          else:
              tmp.append(word[i])
      tmp[len(tmp)-1] = " "

print("".join(tmp))