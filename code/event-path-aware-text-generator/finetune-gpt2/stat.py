import json

def stat():
    e2i, i2e = json.load(open('atomic_event.json', 'r'))

    r = set()

    for e in e2i:
        r.add(e)

    print(len(e2i))
    print(len(r))

word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "'d": " would",
                  "'re": " are", "'ll": " will", "i'm": "i am", "I'm": "I am", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "'ve": " have",  "i've": "i have", "we've": "we have", "can't": "cannot",
                  "aren't": "are not", "isn't": "is not", "wasn't": "was not", "weren't": "were not", "won't": "will not", "there's": "there is",
              "there're": "there are"}

def clean(sentence, word_pairs):
    # sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    # sentence = sentence.replace('personx', 'PersonX')
    # sentence = sentence.replace('persony', 'PersonY')
    # sentence = sentence.replace('personz', 'PersonZ')

    return sentence


def unpack_abbreviation():
    test = open('test.txt', 'r')
    new_test=open('test.txt', 'w')
    for line in test.readlines():
        line = clean(line, word_pairs)
        new_test.write(line)

    dev = open('dev.txt', 'r')
    new_dev = open('dev.txt', 'w')
    for line in dev.readlines():
        line = clean(line, word_pairs)
        new_dev.write(line)

    train = open('train.txt', 'r')
    new_train = open('train.txt', 'w')
    for line in train.readlines():
        line = clean(line, word_pairs)
        new_train.write(line)


def extract_part_path():
    test = open('test.txt', 'r')
    new_test = open('test.txt', 'w')
    i=0
    for line in test.readlines():
        new_test.write(line)
        i+=1
        if i == 5000:
            break

    test = open('train.txt', 'r')
    new_test = open('train.txt', 'w')
    i = 0
    for line in test.readlines():
        new_test.write(line)
        i += 1
        if i == 100000:
            break

    test = open('dev.txt', 'r')
    new_test = open('dev.txt', 'w')
    i = 0
    for line in test.readlines():
        new_test.write(line)
        i += 1
        if i == 5000:
            break


if __name__ == '__main__':
    # stat()
    unpack_abbreviation()
    # extract_part_path()
