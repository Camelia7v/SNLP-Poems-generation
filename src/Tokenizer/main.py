import src.Scraper
from Tokenizer import Tokenizer
import os
import pickle


def get_all_poems():
    authors = []
    poems = []
    corpus = []
    for f in os.listdir('../src/Scraper/Poems'):
        authors.append('../src/Scraper/Poems/' + f)

    for author in authors:
        for f in os.listdir(author):
            if f[-4:] == '.txt':
                poems.append(author + '/' + f)
    for poem in poems:
        with open(poem, 'r', encoding='utf-8') as file:
            lines = file.read()
            corpus.append(lines)
    return corpus


if __name__ == "__main__":
    # with open("./Scraper/Poems/adrian-paunescu/a-mea.txt", 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    corpus = get_all_poems()
    tokenizerr = Tokenizer(corpus)
    tokenizerr.tokenize()
    removed_spaces_corpus = tokenizerr.remove_spaces()
    new_corpus = []
    for array in removed_spaces_corpus:
        text = array.append('<end>')
        text = array.insert(0,'<start>')
        new_corpus.append(text)

    with open("tokenizer.pickle", 'wb') as pickle_file:
        pickle.dump(removed_spaces_corpus, pickle_file)
    # with open("tokenizer.pickle", 'rb') as pickle_file:
    #     pickle_filesss = pickle.load(pickle_file)

