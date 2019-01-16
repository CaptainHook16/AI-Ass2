import operator
import math


class Node:
    def __init__(self, attribute=None, tag=None):
        self.tag = tag
        self.next = {}
        self.attribute = attribute



    def add_node_to_tree(self, data, node_to_add):
        self.next[data] = node_to_add


class Tree:
    def __init__(self, examples, values):
        self.values = values
        self.root = self.ID3_Algorithm(examples, values.keys(), self.compute_best_tag(examples))

    def makeTreeID3Prediction(self, cure_example_test):

        cure_node = self.root
        while cure_node.tag == None:
            cure_node = cure_node.next[cure_example_test[0][cure_node.attribute]]
        return cure_node.tag

    def ID3_Algorithm(self, examples, attributes, default):
        # ID3 algorithm
        if len(examples) == 0:
            return Node(tag=default)
        if self.deal_with_equal_tagging(examples):
            return Node(tag=examples[0][1])
        if len(attributes) == 0:
            return Node(tag=self.compute_best_tag(examples))

        best = self.pick_most_dominante_attribiute(examples, attributes)
        tree = Node(attribute=best)
        for v_i in self.values[best]:
            examples_i = [(inputs, label) for inputs, label in examples if inputs[best] == v_i]
            subtree = self.ID3_Algorithm(examples_i, [a for a in attributes if a != best], self.compute_best_tag(examples))
            tree.add_node_to_tree(v_i, subtree)

        return tree

    def pick_most_dominante_attribiute(self, examples, attributes):

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

        frequency_dict = {}
        length_exmples = len(examples)
        for _, case in examples:
            if case in frequency_dict:
                #if it's already in dict
                frequency_dict[case] =  frequency_dict[case]+ 1
            else:
                #create it
                frequency_dict[case] = 1
        frequency_dict = {l: float(f) / length_exmples for l, f in frequency_dict.items()}
        return sum([-1.0 * prob * math.log(prob) / math.log(2) if prob != 0 else 0 for prob in frequency_dict.values()])

    def deal_with_equal_tagging(self, examples):

        base = examples[0][1]
        for ex in examples:
            if base != ex[1]:
                return False
        return True

    def compute_best_tag(self, examples):

        counts = {}
        for _, l in examples:
            if l in counts:
                counts[l] += 1
            else:
                counts[l] = 0
        return max(counts.items(), key=operator.itemgetter(1))[0]




