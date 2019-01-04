from collections import defaultdict

def max_probability(probs):
    return max(probs,key=lambda item: item[1])[0]

def predict(test_case,train,attributes,target,vocabulary):
    classification_options = attribute_options(train,attributes,target)
    classification_options_frequency = {}
    vocabulary2 = vocabulary.copy()
    for val in classification_options:
        classification_options_frequency[val]=vocabulary2[val]
        del vocabulary2[val]
    print(classification_options_frequency)
    print(vocabulary2)
    index_target = attributes.index(target)
    probs = []
    for tag in classification_options:
        prob_tag = classification_options_frequency[tag]/len(train)
        for index_attribute, attribute in enumerate(test_case):
            #refer only to the rows how contain cure attribute
            #for example - only children age
            attributes_rows = []
            for row in train:
                if row[index_attribute] == attribute:
                    attributes_rows.append(row)
            count=0
            for row in attributes_rows:
                if row[index_target] == tag:
                    count +=1
            prob_tag *= (count+1)/(classification_options_frequency[tag]+vocabulary2[attribute])
        probs.append((tag,prob_tag))

    return max_probability(probs)



def attribute_options(data,attributes,attribute):
    attribute_index = attributes.index(attribute)
    options= set()
    for row in data:
        options.add(row[attribute_index])
    return options


def get_count_for_each_attributes(train):
    count_dict = defaultdict(int)

    for train_example in train:
        for attribute in train_example:
            count_dict[attribute] += 1
    return count_dict



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

    return training_set,test_set,correct_tags,attributes

if __name__ == '__main__':
    training_set, test_set, correct_tags,attributes = load_datasets()
    target = attributes[-1]
    vocabulary = get_count_for_each_attributes(training_set)
    print(vocabulary)
    #predict(None, training_set, attributes, target, vocabulary)

    predictions = []
    for case in test_set:
        pred = predict(case, training_set, attributes, target, vocabulary)
        predictions.append(pred)

    number_of_correct = 0.
    accuracy =0.
    for pred, y in zip(predictions, correct_tags):
        if (pred == y):
            number_of_correct += 1
    accuracy = number_of_correct / len(predictions)
    print('the accuracy is: {}'.format(accuracy))