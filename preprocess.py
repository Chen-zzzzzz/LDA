import numpy as np

def preprocess(filepath="./data/nyt_data.txt", N_lower=150, N_full=200):
    with open(filepath, 'r') as fp:
        inputs = fp.readlines()
    
    valid_article = []
    for i in range(len(inputs)):
        article_raw = []
        vocab_freq = inputs[i].strip('\n').split(',')
        for v in vocab_freq:
            pairs = v.split(':')
            assert len(pairs) == 2
            article_raw.extend([int(pairs[0])] * int(pairs[1]))
        if len(article_raw) < N_lower:
            continue
        else:
            article_raw = np.array(article_raw)
            if len(article_raw) < N_full:
                article_sampled = np.random.choice(article_raw, N_full, replace=True)
            else:
                article_sampled = np.random.choice(article_raw, N_full, replace=False)
            valid_article.append(article_sampled)
    return np.array(valid_article)

def preprocess_custom_article(filepath='./data/article_4.txt', vocabulary="./data/nyt_vocab.txt"):
    N = 200
    with open(vocabulary, 'r') as fp:
        inputs = fp.readlines()
        vocabulary = [val.strip('\n') for val in inputs]
    
    with open(filepath, 'r') as fp:
        inputs = fp.readlines()
    
    article_raw = inputs[0].strip('\n').lower().split(' ')
    # print(len(article_raw), article_raw)
    article_processed = [val.strip(' ').strip('\"').strip('\'').strip('.').strip(',').strip('!').strip('?') for val in article_raw]
    # print(article_processed)
    word_count = 0
    word_freq_dict = {}
    word_list = []
    for w in article_raw:
        if w in vocabulary:
            word_count += 1
            word_freq_dict.setdefault(w, 0)
            word_freq_dict[w] += 1
            idx = article_raw.index(w)
            word_list.append(idx)
    print(word_count, word_freq_dict, word_list)
    if word_count < N:
        word_sampled = np.random.choice(word_list, N, replace=True)
    else:
        word_sampled = np.random.choice(word_list, N, replace=False)

    return word_sampled

def combine_preprocess():
    article_train = preprocess()
    article_custom = np.expand_dims(preprocess_custom_article(), 0)
    article_all = np.concatenate([article_train, article_custom], axis=0)
    return article_all

if __name__ == "__main__":
    a = preprocess()
    print(a.shape)
    b = np.expand_dims(preprocess_custom_article(filepath='./data/article_4.txt'), 0)
    print(b.shape)
    print(np.concatenate([a, b], axis=0).shape)
