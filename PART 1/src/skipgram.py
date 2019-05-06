from config import config
import tensorflow as tf
import numpy as np
import processing
from visualization import project2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '3'}


class SkipGramModel:
    """ 
    Skip gram model 
    """

    def __init__(self, lenVocab, batchSize, dimWordEmbedding, numSamples, learningRate):
        self.lenVocab = lenVocab
        self.batchSize = batchSize
        self.dimWordEmbedding = dimWordEmbedding
        self.numSamples = numSamples
        self.learningRate = learningRate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _dataBatch(self):
        self.centers = tf.placeholder(
            tf.int32, shape=[self.batchSize], name="centers")
        self.targets = tf.placeholder(
            tf.int32, shape=[self.batchSize, 1], name="tagets")

    def _embeddingMaxtrix(self):
        with tf.name_scope('embeddingMatrix'):
            self.embeddingMatrix = tf.Variable(tf.random_uniform(
                [self.lenVocab, self.dimWordEmbedding], -1.0, 1.0), name="embeddingMatrix")

    def _lossFunction(self):
        with tf.name_scope('lossFunction'):
            embedding = tf.nn.embedding_lookup(self.embeddingMatrix, self.centers, name='embedding')
            nce_weight = tf.Variable(tf.truncated_normal(
                            [self.lenVocab, self.dimWordEmbedding], stddev= 1 / (self.dimWordEmbedding)**0.5), 
                            name = 'nce_weights')
            nce_bias = tf.Variable(tf.zeros([self.lenVocab]), name = 'nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight,
                                                        biases = nce_bias, 
                                                        labels = self.targets,
                                                        inputs = embedding,
                                                        num_sampled = self.numSamples,
                                                        num_classes = self.lenVocab), name = 'loss')

    def _optimizer(self):
        with tf.name_scope('optimizer'):
            opm = tf.train.RMSPropOptimizer(self.learningRate)
            self.optimizer = opm.minimize(self.loss, global_step=self.global_step)

    def _summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def buildGraph(self):
        '''
        Activate all the functions above
        '''
        self._dataBatch()
        self._embeddingMaxtrix()
        self._lossFunction()
        self._optimizer()
        self._summaries()


def training(model, batchGenerator, numIteration, vocab):
    
    # training process
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        totalLoss = 0.
        
        for step in range(numIteration):
            centers, targets = next(batchGenerator)
            feed_dict = {model.centers: centers, model.targets: targets}
            lossCumulate, _ = sess.run([model.loss, model.optimizer], 
                                                feed_dict=feed_dict)
            totalLoss += lossCumulate
            if (step + 1) % config.logStep == 0:
                print('Model Loss {}: {:.2f}'.format(step + 1, totalLoss / config.logStep))
                totalLoss = 0.
                returnValue = np.array(sess.run(model.embeddingMatrix))
                project2D(returnValue, vocab, 'TSNE', 'Skip_Gram', step+1)
        return returnValue


def skip_gram():
    vocab = processing.getVocab()
    model = SkipGramModel(config.numVocab, config.batchSize,
                          config.dimWordEmbedding, config.numSamples, config.learningRate)
    model.buildGraph()
    batchGenerator = processing.dataOutput()
    
    vocab = [word for word in vocab.keys()]
    _ = training(model, batchGenerator, config.numIteration, vocab)

    # visualizing the word embedding
    # project2D(embeddingMatrix, vocab, 'TSNE')

def main():
    vocab = processing.getVocab()
    model = SkipGramModel(config.numVocab, config.batchSize,
                          config.dimWordEmbedding, config.numSamples, config.learningRate)
    model.buildGraph()
    batchGenerator = processing.dataOutput()
    
    vocab = [word for word in vocab.keys()]
    _ = training(model, batchGenerator, config.numIteration, vocab)
    
    # print(type(embeddingMatrix))
    # vocab = [word for word in vocab.keys()]
    
    # visualizing the word embedding
    # project2D(embeddingMatrix, vocab, 'TSNE', 'Skip_Gram')

if __name__ == '__main__':
    main()
