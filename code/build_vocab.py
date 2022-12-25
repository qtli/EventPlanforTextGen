
import pdb

class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

        self.word2count['<eos>'] = 9999999999

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

init_index2word = {0: '<eos>'}
lang = Lang(init_index2word)

f1 = open('wiki.train.tokens', 'r')
# f2 = open('valid.fo1', 'r')

for line in f1.readlines():
    line = line.strip().strip('\n')
    lang.index_words(line)


w2c = sorted(lang.word2count.items(), key=lambda x:x[1], reverse=True)
w2c = dict(w2c)
vocab_file = open('vocab.txt', 'w')
idx = 0
for w in w2c:
    vocab_file.write(w + ' ' + str(idx) + '\n')
    idx += 1
