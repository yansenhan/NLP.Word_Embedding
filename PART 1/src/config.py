'''
Function: This file is used to define the config of the skip-gram model
Author: Yansen Han
Date: 03 / 03 / 2019
'''

class config:
    '''
    Set the parameters for all of the data.
    '''

    filePath = '../data/text8.zip'

    dimWordEmbedding = 100
    numVocab = 10000
    lenWindow = 3
    learningRate = 0.1
    batchSize = 128
    numIteration = 50000
    logStep = 2000
    numSamples = 64 # negative samples

    ### glove
    scalingFactor = 0.75
    minFreq = 2
    maxCooccurrence = 100
