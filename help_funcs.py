import math
import operator

# F2I = {}    # maps features to indexes
# I2F = {}    # maps indexes to features
# L2I = {}    # maps classification-labels to indexes
# I2L = {}    # maps indexes to classification-labels
# V2I = {}    # maps feature-values to indexes
# I2V = {}    # maps indexes to feature-values

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
    t = open("output_tree.txt", 'w')
    next = [( I2V[root.attribute][value] , node ) for value,node in root.next.items()]
    # alphabetically sorting by value
    next.sort(key=operator.itemgetter(0))

    for value,n in next:
        if n.tag is None:
            t.write("{}{}={}\n".format(tabs, I2F[root.attribute], value))
            print_tree(n, tabs.split('|')[0] + "\t|")
        else:
            t.write("{}{}={}:{}\n".format(tabs, I2F[root.attribute], value, I2L[n.tag]))

    t.close()


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

