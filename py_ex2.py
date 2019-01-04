from collections import defaultdict

class KNN:
    def __init__(self,training_set=[],test_set=[],correct_predictions=[],k=5):
        self.predictions = []
        self.accuracy = None
        self.train_set = training_set
        self.test_set = test_set
        self.correct_tags = correct_predictions
        self.k = k

    def CalcHammingDistance(self,train_row,test_row):
        """

        :param train_row:
        :param test_row:
        :return:
        """
        sum = 0
        for test_col,train_col in zip(test_row,train_row):
            if(test_col != train_col):
                sum += 1
        return sum

    def computeAccuracy(self):
        number_of_correct = 0.

        for pred, y in zip(self.predictions,self.correct_tags):
            if(pred == y):
                number_of_correct += 1
        self.accuracy =number_of_correct/len(self.predictions)
        print('the accuracy is: {}'.format(self.accuracy))
        #return self.accuracy


    def PredictTag(self,test_case):
        """

        :param test_case:
        :param k:
        :return:
        """

        #first we need to find the most k nearest elements:
        dist = []
        for neighbor_case in self.train_set:
            dist.append((neighbor_case[-1],self.CalcHammingDistance(neighbor_case,test_case)))

        #sort array of distances according to second col - which represents the distance value
        sort_dist = sorted(dist,key=lambda item: item[1])
        k_nearest_neighbors = []
        count = 0
        for index,neighbor in enumerate(sort_dist):
            if index<self.k:
                k_nearest_neighbors.append(neighbor)

        frequency_of_survive = defaultdict(int)
        for neighbor in k_nearest_neighbors:
            frequency_of_survive[neighbor] += 1

        #return the case of most frequent
        yes_or_no = max(frequency_of_survive.items(),key=lambda item: item[1])[0][0]
        return yes_or_no

    def runKnn(self):
        for case in self.test_set:
            pred = self.PredictTag(case)
            self.predictions.append(pred)
        self.computeAccuracy()




def load_datasets():
    training_set = []
    correct_tags = []
    test_set = []
    with open('train.txt') as train_file:
        for line in train_file.readlines()[1:]:
            training_set.append(tuple(line.split()))

    with open('test.txt') as test_file:
        for line in test_file.readlines()[1:]:
            line_cols = line.split()
            correct_tags.append(line_cols[-1])
            test_set.append(tuple(line_cols[:-1]))

    return training_set,test_set,correct_tags



if __name__ == '__main__':
    training_set, test_set, correct_tags = load_datasets()
    knn_model = KNN(training_set,test_set,correct_tags,5)
    knn_model.runKnn()
    print(knn_model.predictions)
