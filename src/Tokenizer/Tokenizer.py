import nltk


class Tokenizer:

    def __init__(self, corpus):
        self.corpus = corpus
        self.nltk_array = []

    def tokenize(self):
        for line in self.corpus:
            nltk_tokens = nltk.word_tokenize(line)
            self.nltk_array.append(nltk_tokens)
        return self.nltk_array

    def remove_spaces(self):
        for line in self.nltk_array:
            if len(line) == 0:
                self.nltk_array.pop(self.nltk_array.index(line))
        return self.nltk_array

