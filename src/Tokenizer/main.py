import src.Scraper
from Tokenizer import Tokenizer
from Mapper import Mapper
import os
import pickle


## Function to get all the poems from the Poems folder into a single corpus
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


## Main function where we tokenize and map words to ids
if __name__ == "__main__":

    ## Getting the corpus from the Poems folder
    corpus = get_all_poems()

    ## Using the Tokenizer class to tokenize sentences.
    tokenizerr = Tokenizer(corpus)
    tokenizerr.tokenize()
    removed_spaces_corpus = tokenizerr.remove_spaces()

    ## Making a new corpus where we add the <start> and <end> token before and after every poem
    new_corpus = []
    for array in removed_spaces_corpus:
        array.append('<end>')
        array.insert(0, '<start>')
        new_corpus.append(array)

    ## Saving the tokenizer to a pickle file.
    with open("tokenizer.pickle", 'wb') as tokenizer_file:
        pickle.dump(new_corpus, tokenizer_file)
    with open("tokenizer.pickle", 'rb') as tokenizer_file:
        tokenizer_results = pickle.load(tokenizer_file)

    print(tokenizer_results)

    ## Using the Mapper class to map each token to an unique id
    mapper = Mapper(new_corpus)
    mapped_corpus = mapper.mapping_word_to_id()

    ## Saving the word to id mapper to a pickle file.
    with open("mapper_word_to_id.pickle", 'wb') as mapper_word_to_id_file:
        pickle.dump(mapped_corpus, mapper_word_to_id_file)
    with open("mapper_word_to_id.pickle", 'rb') as mapper_word_to_id_file:
        mapper_word_to_id_file_results = pickle.load(mapper_word_to_id_file)

    print(mapper_word_to_id_file_results)

    ## Mapper id to word
    # unmapped_corpus = mapper.mapping_id_to_word()
    #
    # ## Saving the id to word mapper to a pickle file.
    # with open("mapper_id_to_word.pickle", 'wb') as mapper_id_to_word_file:
    #     pickle.dump(unmapped_corpus, mapper_id_to_word_file)
    # with open("mapper_id_to_word.pickle", 'rb') as mapper_id_to_word_file:
    #     mapper_id_to_word_file_results = pickle.load(mapper_id_to_word_file)
    #
    # print(mapper_id_to_word_file_results)
