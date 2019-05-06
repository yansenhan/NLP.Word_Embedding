'''
Function: This file is used to process the text file and it will define a iterator for getting samples.
Author: Yansen Han
Date: 03 / 03 / 2019
'''

from config import config
import zipfile
import tensorflow as tf
from nltk.corpus import stopwords
import numpy as np
from collections import Counter, defaultdict

def readData():
    '''
    Read the text file in format of .zip
    '''
    with zipfile.ZipFile(config.filePath) as f:
        # here tf.compat.as_str is used to transform each data within the list into string
        # rather than b'str'
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def vocabConstructor(words, numVocab):
    '''
    Construct a vocabulary and the words is arranged by descending frequency.
    Return a dict {words: index}
    '''
    vocab = dict()
    temp = [('UNK', -1)]
    temp.extend(Counter(words).most_common(numVocab - 1))
    index = 0
    # we don't use 'write' at present, because we don't know how it works.
    for word, _ in temp:
        vocab[word] = index
        index += 1
    return vocab

def words2Index(words, vocab):
    '''
    This function is to convert the entries in words into their index in 'vocab'.
    return a index list
    '''
    return [vocab[word] if word in vocab.keys() else 0 for word in words]

def sampleGenerator(indexWords, windowSize):
    '''
    Require: windowSize need to be odd
    Define an iterator to get samples.
    The output is the index of the original word.
    '''
    if windowSize % 2 == 0:

        windowSize = max(3, windowSize - 1)
        print('windowSize has changed to', windowSize - 1)

    halfWindowSize = windowSize // 2
    for idx, center in enumerate(indexWords):
        for target in indexWords[max(0, idx - halfWindowSize) : idx]:
            yield center, target
        for target in indexWords[idx + 1 : idx + halfWindowSize + 1]:
            yield center, target

def getBatch(iterator, batchSize):
    '''
    Get a batch of data
    Return an array
    '''
    while True:
        center_batch = np.zeros([batchSize], dtype=np.int32)
        target_batch = np.zeros([batchSize, 1])
        for ind in range(batchSize):
            center_batch[ind], target_batch[ind] = next(iterator)
        yield center_batch, target_batch

def dataOutput():
    words = readData()
    vocab = vocabConstructor(words, config.numVocab)
    indexWords = words2Index(words, vocab)
    del words, vocab
    iterator = sampleGenerator(indexWords, config.lenWindow)
    dataBatch = getBatch(iterator, config.batchSize)
    return dataBatch

def getVocab():
    words = readData()
    vocab = vocabConstructor(words, config.numVocab)
    return vocab

####################
# design for glove #
####################

def getPartData(words, position, lenWindow):
    l_window = lenWindow // 2
    r_window = lenWindow // 2
    return words[position - l_window : position + r_window + 1]

def weights4window(words, lenWindow):
    wordCounter = Counter()
    cooccurrenceCounter = defaultdict(float)
    for idx in range(lenWindow // 2, len(words) - lenWindow // 2):
        windowData = getPartData(words, idx, lenWindow)
        wordCounter.update(windowData)
        
        # compute the cooccurrence rate
        word = windowData[lenWindow//2]
        leftContent = windowData[:lenWindow//2]
        rightContent = windowData[lenWindow//2 + 1:]
        for i, context_word in enumerate(leftContent[::-1]):
            cooccurrenceCounter[(word, context_word)] += 1 / (i + 1)
        for i, context_word in enumerate(rightContent):
           cooccurrenceCounter[(word, context_word)] += 1 / (i + 1)
    if len(cooccurrenceCounter) == 0:
            raise ValueError("No cooccurence")
    return wordCounter, cooccurrenceCounter

def defreqWords(wordCounter, numVocab=config.numVocab):
    words = [word for word, _ in wordCounter.most_common(numVocab)]
    return words

def words2Id(words):
    temp = dict()
    i = 0
    for word in words:
        if word in temp:
            continue
        else:
            temp[word] = i
            i += 1
    return temp

def cooccurenceWords2weights(wordsId, words, cooccurrenceCounter):
    cooccurrenceDict = {
        (wordsId[words[0]], wordsId[words[1]]): count
        for words, count in cooccurrenceCounter.items()
        if words[0] in wordsId and words[1] in wordsId}
    return cooccurrenceDict

def getBatch4Glove(lenWindow):
    words = readData()
    print("Finish reading data...")
    wordCounter, cooccurrenceCounter = weights4window(words, lenWindow)
    del words
    print("FInish counting process...")
    words = defreqWords(wordCounter, config.numVocab)
    wordsId = words2Id(words)
    print("Finish defreq words and convert words to id...")
    cooccurrence = cooccurenceWords2weights(wordsId, words, cooccurrenceCounter)
    del wordsId, cooccurrenceCounter, wordCounter
    cooccurrences = [(word_ids[0], word_ids[1], count)
                     for word_ids, count in cooccurrence.items()]
    i_indices, j_indices, counts = zip(*cooccurrences)
    return i_indices, j_indices, counts, words


def getBatchData4Glove(i_indices, j_indices, counts, batchSize, seed):

    np.random.seed = seed
    position = np.random.randint(config.numVocab - batchSize)

    return list(zip(*(i_indices[position : position + batchSize], 
            j_indices[position : position + batchSize], 
            counts[position : position + batchSize])))



