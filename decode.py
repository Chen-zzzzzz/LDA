import numpy as np
from scipy.stats import dirichlet

K = 25
V = 1715

beta = np.load('beta_eco.npy')
gamma = np.load('gamma_eco.npy')
alpha = np.load('alpha_eco.npy')
voc = open('data/nyt_vocab.txt', 'r').readlines()
voc = [i.strip("\n") for i in voc]
for i in range(beta.shape[0]):
    idx = beta[i].argsort(axis=0)[-10:]
    keyword_topic = [voc[idx[j]] for j in range(10)]
    print("Group ", i, keyword_topic)

article_topics = np.argsort(gamma[-1, :])[:]
print(article_topics, gamma.shape)
for topic in article_topics:
    idx = beta[topic].argsort(axis=0)[-10:]
    keyword_topic = [voc[idx[j]] for j in range(10)]
    print("Topics ", topic, keyword_topic)

print('alpha', alpha)

p = dirichlet.mean(gamma[-1, :])
print(p)
topics = np.random.choice(np.arange(K), size=30, p=p)
word_idx = []
for t in topics:
    word_idx.append(np.random.choice(np.arange(V), size=1, p=beta[t])[0])
word_generate = [voc[i] for i in word_idx]
print(word_generate)