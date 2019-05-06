'''
Function: This file is to visualize the word embedding in a 2-D plane.
Author: Yansen Han
Date: 04 / 03 / 2019
'''
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

def project2D(embeddingMatrix, vocab, kind='PCA', model='Skip_Gram', currentIteration=-1):
    '''
    Project the matrix into 2-D plane
    Input:
    --embeddingMatrix: array, shape: (lenVocab, 300)
    --vocab: list, length: lenVocab
    --kind: PCA or TSNE
    '''
    if kind == 'PCA':
        symMatrix = embeddingMatrix.T.dot(embeddingMatrix)
        u, _, _ = svd(symMatrix)
        vec1 = embeddingMatrix.dot(u[:][0])
        vec2 = embeddingMatrix.dot(u[:][1])
        vecs = np.vstack([vec1, vec2]).T

        plt.figure(figsize=(15, 15), dpi=150)
        plt.grid(True, linestyle='--', color='gray')
        for idx, vec in enumerate(vecs[200:300]):
            plt.plot(vec[0], vec[1], 'gx')
            plt.text(vec[0], vec[1], vocab[idx + 200], fontsize=10)
        if currentIteration != -1:
            pic_name = './images_PCA/'+ model +'_Iteration_' \
                        + str(currentIteration) + '_TSNE'
            plt.savefig(pic_name, dpi=150)
            plt.close()
    elif kind == 'TSNE':
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddingMatrix[200:300])
        labels = vocab[200:300]
        plt.figure(figsize=(15, 15), dpi=150)  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                        va='bottom')
            plt.xlabel(model)
        if currentIteration != -1:
            pic_name = './images_TSNE/'+ model +'_Iteration_' \
                        + str(currentIteration) + '_TSNE'
            plt.savefig(pic_name, dpi=150)
            plt.close()