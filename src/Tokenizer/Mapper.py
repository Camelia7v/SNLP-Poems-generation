class Mapper:
    def __init__(self, corpus):
        self.corpus = corpus
        self.mapper_word = {}
        self.mapper_id = {}

    def mapping_word_to_id(self):
        word_id = 0
        for poem in self.corpus:
            for token in poem:
                if token not in self.mapper_word.keys():
                    self.mapper_word[token.lower()] = word_id
                    word_id = word_id + 1
        return self.mapper_word

    def mapping_id_to_word(self):
        word_id = 0
        for poem in self.corpus:
            for token in poem:
                if token not in self.mapper_id.values():
                    self.mapper_id[word_id] = token.lower()
                    word_id = word_id+1
        return self.mapper_id
