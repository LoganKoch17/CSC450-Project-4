"""
HumanStateClassifier.py is the HumanStateClassifier component referenced in section 3.2 of the SDD, A03_SDD_Team4.docx. 



Copyright (c) 2020 Fall Detection System, All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.	Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
import numpy as np
from timeit import default_timer as timer

# Allows the use of a max heap in python.
# This is the predicted optimal datastrucutre to contain the neighbors calculated Euclidean distance measure as
# we would only extract the highest value O(1), insert a new value (log(n)) or extract the max value (log(n)).
# NOTE: Heapq is noramlly implements a heap as a min heap, to use it as a max heap, 
# all values will be multiplied by -1
from heapq import heappop, heappush, heapify


"""
DistanceAndClass referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 

"""
class DistanceAndClass:
    def __init__(self, distance, classification):
        self.distance = distance
        self.classification = classification
    
    # allows heapq to compare the DistanceAndClass object to another
    def __lt__(self, other):
        return self.distance < other.distance
    
    def __str__(self):
        return (f"{self.distance},{self.classification}")

"""
KNeighborsClassifier referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 
Max Heap referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx, and is utilized in the KNeighborsClassifier Class.

The Functional Requirement:

FR.4 The system must determine if an object classified as “human” is lying or sitting down.
FR.5 The system must be able to differentiate between when a person falls down and when they go to sit or lay down of their own will.

is addressed in the classify() method of the KNeighborsClassifier class.
"""
class KNeighborsClassifier:
    def __init__(self, trainingDataset, k = None):
        self.training_dataset = trainingDataset

        if k is not None:
            self.k = k
        else:
            self.k = 5

    def set_training_dataset(self, trainingData):
        self.training_dataset = trainingData

    def set_k(self, val):
        self.k = val

    # K Nearest Neighbors algorithm
    def classify(self, testing_item):
        classification = None
        neighbors = []
        heapify(neighbors)

        for classification_set in self.training_dataset.items():
            classification = classification_set[0]
            for training_item in classification_set[1]:
                sum_euclidean = self.euclidean_distance(testing_item, training_item)
                distance_and_class = DistanceAndClass(-1 * sum_euclidean, classification)
                if len(neighbors) < self.k:
                    heappush(neighbors, distance_and_class)
                elif (-1 * distance_and_class.distance < neighbors[0].distance * -1):
                    heappop(neighbors)
                    heappush(neighbors, distance_and_class)

        classifiers = {}
        for neighbor in neighbors:
            neighbor_class = neighbor.classification
            try:
                weighted_vote = 1.0 / neighbor.distance * -1
            except(ZeroDivisionError) as error:
                weighted_vote = 1.0
            if neighbor_class in classifiers:
                classifiers[neighbor_class] += weighted_vote
            else:
                classifiers[neighbor_class] = weighted_vote

        largest_vote = max(classifiers.values())

        if largest_vote < .0008:
            classification = "unrecognized"
        else:
            classification = max(classifiers, key = classifiers.get)
        

        return classification

    def euclidean_distance(self, source, target):
        if source is not None and target is not None:
            distance = np.linalg.norm(source - target)
            return distance
        else:
            return 1