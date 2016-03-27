import json
import nltk
from nltk.probability import FreqDist

data_path = './authorship-verification-dataset/'

with open(data_path + 'contents.json') as data_file:    
    data = json.load(data_file)

sample_names = data['problems']

samples = [[]]
samples.append([])
samples.append([])

for sample_name in sample_names:
    with open(data_path +  sample_name + '/known01.txt') as known:
        samples[0].append(nltk.word_tokenize(known.read().replace('\n', '').strip()))
    with open(data_path +  sample_name + '/unknown.txt', 'r') as unknown:
        samples[1].append(nltk.word_tokenize(unknown.read().replace('\n', '').strip()))

words_count = len(samples[0][0])
print words_count

word_categories = ['the','a','.','!','?',':',';']
tag_categories = ['CC', 'CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS',
                   'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO',
                   'UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']



frequencies = [[]]

for i in range(len(samples[0])):
    words_frequency = FreqDist([w.lower() for w in samples[0][i]])
    tagged_words = nltk.pos_tag(samples[0][i])
    tags = list(map(lambda x : x[1], tagged_words))
    tags_frequency = FreqDist(tags)
    
    for word_category in word_categories:
        frequencies[len(frequencies) - 1].append(words_frequency[word_category])
    for tag_category in tag_categories:
        frequencies[len(frequencies) - 1].append(tags_frequency[tag_category])
    frequencies.append([])
    
print frequencies


# tags_bigrams = list(nltk.bigrams(tags))
# tags_bigrams_frequency = FreqDist(tags_bigrams)
# print tags_bigrams_frequency.items()

# tag_frequency = FreqDist(tags)
# print tags_frequency.items()


# print len(tags_categories)