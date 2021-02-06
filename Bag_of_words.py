import numpy as np
class Bag_of_words:
    def __init__(self):
        self.cuvinte = []
        self.vocabulary_len = 0

    def build_vocabulary(self, data):
        for str in data:
               str = str.split()
               for cuvant in str:
                    if cuvant not in self.cuvinte:
                        self.cuvinte.append(cuvant)

        self.vocabulary_len = len(self.cuvinte)
        self.cuvinte = np.array(self.cuvinte)

    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_len))
        for document_idx, document in enumerate(data):
            document = document.split()
            for cuvant in document:
                if cuvant in self.cuvinte:
                    idx = np.where(self.cuvinte == cuvant)[0][0]
                    features[document_idx, idx] += 1
        return features