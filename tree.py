import operator
import math


class Node:
    def __init__(self, attribute=None, classification=None):
        self.attribute = attribute
        self.classification = classification
        self.next = {}

    def add_child(self, value, node):
        self.next[value] = node


class Model:
    def __init__(self, examples, values):
        self.values = values
        self.root = self.DTL(examples, values.keys(), self.best_classification(examples))

    def DTL(self, examples, attributes, default):
        # ID3 algorithm
        if len(examples) == 0:
            return Node(classification=default)
        if self.same_classification(examples):
            return Node(classification=examples[0][1])
        if len(attributes) == 0:
            return Node(classification=self.best_classification(examples))

        best = self.choose_attribute(examples, attributes)
        tree = Node(attribute=best)
        for v_i in self.values[best]:
            examples_i = [(inputs, label) for inputs, label in examples if inputs[best] == v_i]
            subtree = self.DTL(examples_i, [a for a in attributes if a != best], self.best_classification(examples))
            tree.add_child(v_i, subtree)

        return tree

    def choose_attribute(self, examples, attributes):
        '''
        chooses the best attribute for the list of examples
        @param examples: list of examples
        @param attributes: list of (available) attributes
        '''
        E = self.entropy(examples)
        gains = {}
        for attribute in attributes:
            part_entropy = 0.0
            for v_i in self.values[attribute]:
                examples_i = [(inputs, label) for inputs, label in examples if inputs[attribute] == v_i]
                p = float(len(examples_i)) / len(examples)
                part_entropy += p * self.entropy(examples_i)
            gains[attribute] = E - part_entropy

        return max(gains.items(), key=operator.itemgetter(1))[0]

    def entropy(self, examples):
        '''
        calculates and returns the entropy on the list of examples
        @param examples: list of examples
        '''
        freq = {}
        for _, label in examples:
            if label in freq:
                freq[label] += 1
            else:
                freq[label] = 1
        freq = {l: float(f) / len(examples) for l, f in freq.items()}
        return sum([-1.0 * p * math.log(p) / math.log(2) if p != 0 else 0 for p in freq.values()])

    def same_classification(self, examples):
        '''
        checks wether all the examples have the same classification
        '''
        base = examples[0][1]
        for ex in examples:
            if base != ex[1]:
                return False
        return True

    def best_classification(self, examples):
        '''
        finds the best classification for the set of examples (majority wins)
        '''
        counts = {}
        for _, l in examples:
            if l in counts:
                counts[l] += 1
            else:
                counts[l] = 0
        return max(counts.items(), key=operator.itemgetter(1))[0]

    def predict(self, inputs):
        '''
        predicting classification on single input
        '''
        c = self.root
        while c.classification == None:
            c = c.next[inputs[0][c.attribute]]
        return c.classification





# import operator
# import random as rd
# import math
# from collections import defaultdict
# import random as random
# class Node:
#     def __init__(self,classification=None,attribute = None):
#         self.tag = classification
#         self.attribute = attribute
#         self.next = {}
#
#     def add_node(self,val,node_to_add):
#         self.next[val] = node_to_add
#
#
# class Tree:
#     def __init__(self,data,attributes,target):
#         #self.options = options
#         self.root = self.ID3(data, attributes, choose_most_class(data,attributes,target),attributes[-1])
#         #self.root = None
#
#     def ID3(self,data,attributes,default_class,target):
#         # default_class = choose_most_class(data,attributes,target)
#         if len(data) == 0:
#             # self.root = Node(classification=default_class)
#             return Node(classification=default_class)
#         target_vals = []
#         target_index = attributes.index(target)
#         for row in data:
#            target_vals.append(row[target_index])
#
#         #if all values are the same return this options
#         if target_vals.count(target_vals[0]) == len(target_vals):
#             # self.root = Node(classification=target_vals[0])
#             return Node(classification=target_vals[0])
#         #if there is no attributes left
#         if len(attributes)==1:
#             # self.root =
#             return  Node(classification=default_class)
#
#         most_dominante_attribute = mostDominanteAttribute(data,target,attributes)
#         new_node = Node(attribute=most_dominante_attribute)
#         best_attributes_options = get_attribute_options(data,most_dominante_attribute,attributes)
#         for option in best_attributes_options:
#             data_by_best_attribute = get_data_by_attribute(option,data,dominante_attribute=most_dominante_attribute,attributes=attributes)
#             attributes_of_best = attributes.copy()
#             attributes_of_best.remove(most_dominante_attribute)
#
#             child_tree = self.ID3(data_by_best_attribute,attributes_of_best,target)
#             new_node.add_node(option,child_tree)
#
#         # self.root=new_node
#         return new_node
#
#     def best_classification(self, examples):
#         '''
#         finds the best classification for the set of examples (majority wins)
#         '''
#         counts = {}
#         for _,l in examples:
#             if l in counts:
#                 counts[l] += 1
#             else:
#                 counts[l] = 0
#         return max(counts.items(), key=operator.itemgetter(1))[0]
#
#     def predict(self,test_case,attributes):
#         cure = self.root
#         while cure.tag == None:
#             index = attributes.index(cure.attribute)
#             cure = cure.next[test_case[index]]
#         return cure.tag
#
#
# def mostDominanteAttribute(data,target,attributes):
#     most_dominante = attributes[0]
#     max_info_gain = 0
#
#     for attribute in attributes:
#         #dont check the targget
#         if target!=attribute:
#             cure_gain = calcGain(data,attribute,attributes,target)
#             if max_info_gain < cure_gain:
#                 max_info_gain = cure_gain
#                 most_dominante = attribute
#     return most_dominante
#
# def get_attribute_options(data,attribute,attributes):
#     attribute_index = attributes.index(attribute)
#     options = set()
#     for row in data:
#         options.add(row[attribute_index])
#     return options
#
# def get_data_by_attribute(option,data,dominante_attribute,attributes):
#     dominante_index = attributes.index(dominante_attribute)
#     data_best_attribute = []
#
#     for row in data:
#
#         #get only row with sunny wether for example
#         if(row[dominante_index] == option):
#             new_row = []
#             new_row = row[:dominante_index]+row[dominante_index+1:]
#             data_best_attribute.append(tuple(new_row))
#     return data_best_attribute
#
# def choose_most_class(data,attributes,target):
#     index_target = attributes.index(target)
#     frequency = defaultdict(int)
#
#     for row in data:
#         frequency[row[index_target]] +=1
#
#     most_class = max(frequency.items(),key=lambda item: item[1])[0]
#     return most_class
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
#     return training_set, test_set, correct_tags, attributes
#
#
# # class DecisionTree:
# #     def __init__(self, train, test):
# #         self.training_set = train
# #         self.test_set = test
#
#
# def entropy(data, chosenAttribute, attributes):
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
#         # for example update the number of survivals and deads
#         frequency_dict[case[index_of_target]] += 1
#
#     labels = frequency_dict.values()
#     entropy_val = 0.
#     for label_freq in labels:
#         prob = label_freq / len(data)
#         entropy_val += -prob * math.log(prob, 2)
#     return entropy_val
#
#
# def remainder(data, checkedAttribute, attributes, target):
#     # for example - checkedAttribute - wether
#     # target - decision
#     index_of_attribute = attributes.index(checkedAttribute)
#     frequency_dict = defaultdict(int)
#
#     for case in data:
#         # for example update the number of survivals and deads
#         frequency_dict[case[index_of_attribute]] += 1
#
#     attribute_entropy = 0.
#
#     for lable in frequency_dict.keys():
#         # number of sunny days for example:
#         prob = frequency_dict[lable] / sum(frequency_dict.values())
#         data_rows_by_lable = []
#         for row in data:
#             if row[index_of_attribute] == lable:
#                 data_rows_by_lable.append(row)
#         attribute_entropy += prob * entropy(data_rows_by_lable, target, attributes)
#     return attribute_entropy
#
#
# def calcGain(data, checkedAttribute, attributes, target):
#     return entropy(data, target, attributes) - remainder(data, checkedAttribute, attributes, target)
#
# def run_ID3(training_set, test_set, correct_tags,attributes):
#     tree = Tree()
#
#     tree.ID3(data=training_set, attributes=attributes, target=attributes[-1])
#
#     predictions = []
#
#     for case in test_set:
#         pred = tree.predict(case, attributes)
#         predictions.append(pred)
#
#     number_of_correct = 0.
#     accuracy = 0
#     for pred, y in zip(predictions, correct_tags):
#         if (pred == y):
#             number_of_correct += 1
#     accuracy = math.ceil(number_of_correct / len(predictions) * 100) / 100
#    # accuracy = number_of_correct / len(predictions)
#     print('the accuracy is: {}'.format(accuracy))
#     return predictions,accuracy
#
# def ID3results():
#     training_set, test_set, correct_tags, attributes = load_datasets()
#     tree = Tree(training_set,attributes,attributes[-1])
#
#     # tree.ID3(data=training_set, attributes=attributes, target=attributes[-1])
#
#     predictions = []
#
#     for case in test_set:
#         pred = tree.predict(case, attributes)
#         predictions.append(pred)
#
#     number_of_correct = 0.
#     accuracy = 0
#     for pred, y in zip(predictions, correct_tags):
#         if (pred == y):
#             number_of_correct += 1
#     accuracy = math.ceil(number_of_correct / len(predictions) * 100) / 100
#     # accuracy = number_of_correct / len(predictions)
#     print('the accuracy is: {}'.format(accuracy))
#     return predictions, accuracy,tree
#     #run_ID3(training_set, test_set, correct_tags, attributes)
#
# if __name__ == '__main__':
#     training_set, test_set, correct_tags,attributes = load_datasets()
#     print("corrects")
#     print(len(correct_tags))
#     tree = Tree()
#     tree.ID3(data=training_set,attributes=attributes,target=attributes[-1])
#
#     predictions = []
#
#
#     for case in test_set:
#         pred = tree.predict(case,attributes)
#         predictions.append(pred)
#
#     number_of_correct = 0.
#     accuracy =0
#     for pred, y in zip(predictions, correct_tags):
#         if (pred == y):
#             number_of_correct += 1
#     accuracy = number_of_correct / len(predictions)
#     print('the accuracy is: {}'.format(accuracy))
