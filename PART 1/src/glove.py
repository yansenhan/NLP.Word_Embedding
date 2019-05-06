from config import config
import tensorflow as tf
import numpy as np
import processing
from visualization import project2D

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '3'}

class GloVe:
    '''
    GloVe model
    '''

    def __init__(self, lenVocab, batchSize, dimWordEmbedding, learningRate, maxCooccurrence, scalingFactor):
        self.lenVocab = lenVocab
        self.batchSize = batchSize
        self.dimWordEmbedding = dimWordEmbedding
        self.learningRate = learningRate
        self.maxCooccurrence = maxCooccurrence
        self.scalingFactor = scalingFactor
        
    def _varInit(self):
        with tf.name_scope('VariableInitialization'):

            self.focal_input = tf.placeholder(
                tf.int32, shape=[self.batchSize], name="focal_words")
            self.context_input = tf.placeholder(
                tf.int32, shape=[self.batchSize], name="context_words")
            self.cooccurrence_count = tf.placeholder(
                tf.float32, shape=[self.batchSize], name="cooccurrence_count")
    
    def _embeddingMatrix(self):
        with tf.name_scope('EmbeddingMatrixs'):
            self.focal_embeddings = tf.Variable(
                tf.random_uniform(
                    [self.lenVocab, self.dimWordEmbedding], 1.0, -1.0),
                name="focal_embeddings")
            self.context_embeddings = tf.Variable(
                tf.random_uniform(
                    [self.lenVocab, self.dimWordEmbedding], 1.0, -1.0),
                name="context_embeddings")

            self.focal_biases = tf.Variable(tf.random_uniform([self.lenVocab], 1.0, -1.0),
                                       name='focal_biases')
            self.context_biases = tf.Variable(tf.random_uniform([self.lenVocab], 1.0, -1.0),
                                         name="context_biases")

    def _lossFunction(self):
        with tf.name_scope('LossFunction'):

            maxCooccurrence = tf.constant([self.maxCooccurrence], dtype=tf.float32,
                                          name='maxCooccurrence')
            scalingFactor = tf.constant([self.scalingFactor], dtype=tf.float32,
                                        name="scalingFactor")

            focal_embedding = tf.nn.embedding_lookup([self.focal_embeddings], self.focal_input)
            context_embedding = tf.nn.embedding_lookup([self.context_embeddings], self.context_input)
            focal_bias = tf.nn.embedding_lookup([self.focal_biases], self.focal_input)
            context_bias = tf.nn.embedding_lookup([self.context_biases], self.context_input)

            weighting_factor = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count, maxCooccurrence), scalingFactor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.loss = tf.reduce_sum(single_losses)

    def _optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdagradOptimizer(self.learningRate).minimize(
                self.loss)
            self.combined_embeddings = tf.add(self.focal_embeddings, self.context_embeddings,
                                                name="combined_embeddings")
    
    def buildGraph(self):
        # self.__graph = tf.Graph()
        # with self.__graph.as_default(), self.__graph.device(_device_for_node):
        with tf.name_scope('Graph'):
            self._varInit()
            self._embeddingMatrix()
            self._lossFunction()
            self._optimizer()

def training(model, numIteration):
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        totalLoss = 0.

        print("Start Iteration...")
        i_indices, j_indices, _counts, words = processing.getBatch4Glove(config.lenWindow)
        for step in range(numIteration):
            batches = processing.getBatchData4Glove(i_indices, j_indices, _counts, model.batchSize, step)
            # shuffle(batches)
            i_s, j_s, counts = list(zip(*batches))
            if len(counts) != model.batchSize:
                continue
            feed_dict = {
                model.focal_input: i_s,
                model.context_input: j_s,
                model.cooccurrence_count: counts}
            lossCumulate, _ = sess.run([model.loss, model.optimizer],
                                        feed_dict=feed_dict)
            totalLoss += lossCumulate
            if (step + 1) % config.logStep == 0:
                print('Model Loss {}: {:.2f}'.format(step + 1, totalLoss / config.logStep))
                totalLoss = 0.
                embeddingMatrix = np.array(sess.run(model.combined_embeddings))
                project2D(embeddingMatrix, words, 'TSNE', 'GloVe', step+1)
        return embeddingMatrix, words

def glove():
    
    model = GloVe(config.numVocab, config.batchSize,
                          config.dimWordEmbedding, config.learningRate, config.maxCooccurrence, config.scalingFactor)
    
    model.buildGraph()
    embeddingMatrix, words = training(model, config.numIteration) 
       
def main():
    
    model = GloVe(config.numVocab, config.batchSize,
                          config.dimWordEmbedding, config.learningRate, config.maxCooccurrence, config.scalingFactor)
    
    model.buildGraph()
    embeddingMatrix, words = training(model, config.numIteration)
    
    # visualizing the word embedding
    # project2D(embeddingMatrix, words, 'TSNE', 'GloVe')

if __name__ == '__main__':
    main()
