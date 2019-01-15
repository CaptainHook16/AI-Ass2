import math
import operator
import random
from collections import defaultdict
import knn as knn_alg
import tree as ID3
import NaiveBayesClassifier as nb_alg
import help_funcs as ut


# global F2I,I2F,I2F,L2I,I2L,V2I,I2V
F2I = {}    # maps features to indexes
I2F = {}    # maps indexes to features
L2I = {}    # maps classification-labels to indexes
I2L = {}    # maps indexes to classification-labels
V2I = {}    # maps feature-values to indexes
I2V = {}    # maps indexes to feature-values

def load_datasets():
    """
    load the datasets the model need
    :return: training_set,test_set,correct_tags
    """
    training_set = []
    correct_tags = []
    test_set = []
    with open('train.txt') as train_file:
        attributes = train_file.readline().split()
        for line in train_file.readlines()[1:]:
            training_set.append(tuple(line.split()))

    with open('test.txt') as test_file:
        for line in test_file.readlines()[1:]:
            line_cols = line.split()
            correct_tags.append(line_cols[-1])
            test_set.append(tuple(line_cols[:-1]))

    # random.shuffle(training_set)

    return training_set,test_set,correct_tags,attributes

def output_predictions(tree_data, knn_data, nb_data, predictions_len, output_file_path):
    """ save all the models predictions to the output file """
    """ tree_data is in format of tuple: (tree_predictions, tree_accuracy) and etc. for each one of them """

    with open(output_file_path, 'w') as f:

        f.write('{}\t{}\t{}\t{}\n'.format('Num', 'DT', 'KNN', 'naiveBase'))
        for index in range(predictions_len):
            f.write('{}\t{}\t{}\t{}\n'.format(index + 1, tree_data[0][index], knn_data[0][index], nb_data[0][index]))

        f.write('{}\t{}\t{}\t{}'.format('', str(tree_data[1]), str(knn_data[1]), str(nb_data[1])))

def pre_process(filename):
    '''
    mapping all the features, their values and classifications to indexes and vice versa
    '''
    global F2I,I2F,L2I,I2L,V2I,I2V
    n = 1
    for line in open(filename):
        l = line.split()
        if n == 1:
            # extracting names of the features
            F2I = {f:i for i,f in enumerate(l[:-1])}
            I2F = {i:f for i,f in enumerate(l[:-1])}
            for i,f in enumerate(l[:-1]):
                V2I[I2F[i]] = {}
        else:
            # extracting the classification-labels
            if l[-1] not in L2I:
                L2I[l[-1]] = len(L2I)
            # extracting the feature-values
            for i,f in enumerate(l[:-1]):
                if f not in V2I[I2F[i]]:
                    V2I[I2F[i]][f] = len(V2I[I2F[i]])
        n += 1

    I2L = {i:l for l,i in L2I.items()}
    I2V = {}
    for f,v2i in V2I.items():
        I2V[F2I[f]] = {i:v for v,i in v2i.items()}

def print_tree(root, tabs=""):
    '''
    printing the tree the DecisionTree algorithm ceated
    @param root: the root of the (sub)tree
    @param tabs: string containing desired tabs
    '''
    # retrieving tuples of value:node
    next = [( I2V[root.attribute][value] , node ) for value,node in root.next.items()]
    # alphabetically sorting by value
    next.sort(key=operator.itemgetter(0))

    for value,n in next:
        if n.classification is None:
            t.write("{}{}={}\n".format(tabs, I2F[root.attribute], value))
            print_tree(n, tabs.split('|')[0] + "\t|")
        else:
            t.write("{}{}={}:{}\n".format(tabs, I2F[root.attribute], value, I2L[n.classification]))



def get_data(filename):
    '''
    reading the data from a given file
    @param filename: name of the file
    '''
    data = []

    n = 1
    for line in open(filename):
        if n == 1:
            n += 1
            continue
        l = line.split()
        x = [V2I[I2F[i]][f] for i,f in enumerate(l[:-1])]
        y = L2I[l[-1]]
        data.append( (x , y) )

    return data


def predictTree(test_set,treeAlg):
    predictions= []
    accuracy = 0.
    corrects_len=0
    for i,cure_example in enumerate(test_set):
        y_prediction =treeAlg.predict(cure_example)
        if y_prediction==cure_example[1]:
            corrects_len +=1
        predictions.append(y_prediction)
    accuracy = math.ceil(corrects_len / len(predictions) * 100) / 100
    return predictions,accuracy





if __name__ == '__main__':
    pre_process("train.txt")

    # print(F2I)
    # print(I2F)
    # print(L2I)
    # print(I2L)
    # print(V2I)
    # print(I2V)

    TRAIN = get_data("train.txt")
    TEST = get_data("test.txt")
    print("TESTTTTTTTTTTT")
    print(TEST)

    # creating the different models
    dt = ID3.Model(TRAIN, values={i: value.keys() for i, value in I2V.items()})

    predsKNN,accuracyKNN,corrects = knn_alg.runKnn()
    predsNB, accuracyNB = nb_alg.NBresults()
    predsID3, accuracyID3 = predictTree(test_set=TEST,treeAlg=dt)



    output_predictions((predsID3,accuracyID3),(predsKNN,accuracyKNN),(predsNB,accuracyNB),len(corrects),'output.txt')

    # printing the tree that DecisionTree created
    t = open("output_tree.txt", 'w')
    print_tree(dt.root)
    t.close()


    # training_set, test_set, correct_tags, attributes = load_datasets()
    #preds_Naive_bayes = nb_alg.run_naive_bayes(training_set, test_set, correct_tags, attributes)
    # print(preds_Naive_bayes)
    # id3_preds = ID3.run_ID3(training_set, test_set, correct_tags, attributes)
    # print(id3_preds)
    # knn_model = knn_alg.KNN(training_set, test_set, correct_tags, 5)
    # knn_model.runKnn()
    # print(knn_model.predictions,knn_model.accuracy)


#
# class KNN:
#     """
#     KNN class - implements the k-nearest neighbors algorithm
#     """
#     def __init__(self,training_set=[],test_set=[],correct_predictions=[],k=5):
#         self.predictions = []
#         self.accuracy = None
#         self.train_set = training_set
#         self.test_set = test_set
#         self.correct_tags = correct_predictions
#         self.k = k
#
#     def CalcHammingDistance(self,train_row,test_row):
#
#         """
#
#         :param train_row:
#         :param test_row:
#         :return:the distance of the two samples
#         """
#         sum = 0
#         for test_col,train_col in zip(test_row,train_row):
#             if(test_col != train_col):
#                 sum += 1
#         return sum
#
#     def computeAccuracy(self):
#         """
#         the function compute the accuracy of the model by checking how many
#         cases the model succeed to predict correct
#
#         :return:None
#         """
#         number_of_correct = 0.
#
#         for pred, y in zip(self.predictions,self.correct_tags):
#             if(pred == y):
#                 number_of_correct += 1
#         self.accuracy =number_of_correct/len(self.predictions)
#         print('the accuracy is: {}'.format(self.accuracy))
#         #return self.accuracy
#
#
#     def PredictTag(self,test_case):
#         """
#
#         :param test_case:
#         :param k:
#         :return:the model predict if this case result is survive/not
#         """
#
#         #first we need to find the most k nearest elements:
#         dist = []
#         for neighbor_case in self.train_set:
#             dist.append((neighbor_case[-1],self.CalcHammingDistance(neighbor_case,test_case)))
#
#         #sort array of distances according to second col - which represents the distance value
#         sort_dist = sorted(dist,key=lambda item: item[1])
#         k_nearest_neighbors = []
#         count = 0
#         for index,neighbor in enumerate(sort_dist):
#             if index<self.k:
#                 k_nearest_neighbors.append(neighbor)
#
#         frequency_of_survive = defaultdict(int)
#         for neighbor in k_nearest_neighbors:
#             frequency_of_survive[neighbor] += 1
#
#         #return the case of most frequent
#         yes_or_no = max(frequency_of_survive.items(),key=lambda item: item[1])[0][0]
#         return yes_or_no
#
#     def runKnn(self):
#         for case in self.test_set:
#             pred = self.PredictTag(case)
#             self.predictions.append(pred)
#         self.computeAccuracy()
#
#
#
#
# def load_datasets():
#     """
#     load the datasets the model need
#     :return: training_set,test_set,correct_tags
#     """
#     training_set = []
#     correct_tags = []
#     test_set = []
#     with open('train.txt') as train_file:
#         attributes = train_file.readline().split()
#         for line in train_file.readlines()[1:]:
#             training_set.append(tuple(line.split()))
#
#     with open('test.txt') as test_file:
#         for line in test_file.readlines()[1:]:
#             line_cols = line.split()
#             correct_tags.append(line_cols[-1])
#             test_set.append(tuple(line_cols[:-1]))
#
#     return training_set,test_set,correct_tags,attributes
#
# class DecisionTree:
#     def __init__(self,train,test):
#         self.training_set = train
#         self.test_set = test
#
# def entropy(data,chosenAttribute,attributes):
#     """
#
#     :param data:
#     :param chosenAttribute:
#     :param atrributes:
#     :return:
#     """
#     index_of_target = attributes.index(chosenAttribute)
#     frequency_dict = defaultdict(int)
#
#     for case in data:
#         #for example update the number of survivals and deads
#         frequency_dict[case[index_of_target]] +=1
#
#     labels = frequency_dict.values()
#     entropy_val = 0.
#     for label_freq in labels:
#         prob = label_freq/len(data)
#         entropy_val += -prob*math.log(prob,2)
#     return entropy_val
#
# def remainder(data,checkedAttribute,attributes,target):
#     #for example - checkedAttribute - wether
#     #target - decision
#     index_of_attribute = attributes.index(checkedAttribute)
#     frequency_dict = defaultdict(int)
#
#     for case in data:
#         # for example update the number of survivals and deads
#         frequency_dict[case[index_of_attribute]] += 1
#
#     attribute_entropy= 0.
#
#     for lable in frequency_dict.keys():
#         #number of sunny days for example:
#         prob = frequency_dict[lable]/sum(frequency_dict.values())
#         data_rows_by_lable = []
#         for row in data:
#             if row[index_of_attribute] == lable:
#                 data_rows_by_lable.append(row)
#         attribute_entropy += prob*entropy(data_rows_by_lable,target,attributes)
#
# def gain(data,checkedAttribute,attributes,target):
#     return entropy(data,target,attributes) - remainder(data,checkedAttribute,attributes,target)
#
#
#
# if __name__ == '__main__':
#     training_set, test_set, correct_tags,attributes = load_datasets()
#     #create the KNN model with k=5 as required
#     knn_model = KNN(training_set,test_set,correct_tags,5)
#     knn_model.runKnn()
#     print(knn_model.predictions)
#     training_set = []
#
#     # with open('train.txt') as train_file:
#     #     arts = train_file.readline().split()
#     #     for line in train_file.readlines()[1:]:
#     #         training_set.append(tuple(line.split()))
#     # print(arts)
