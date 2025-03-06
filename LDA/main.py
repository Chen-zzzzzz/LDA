import numpy as np
from scipy.special import digamma, polygamma, psi
from LDA.preprocess import *
import time

M = 2910+1  # number of articles
N = 200     # length of articles
K = 25      # number of topics
V = 1715    # vocabulary
# init
phi = np.ones((M, N, K)) / float(K)

alpha = 0.1 * np.ones((K, ))
gamma = np.tile(np.expand_dims(alpha, 0), (M, 1)) + np.ones((M, K)) * N / float(K)

beta = np.random.uniform(size=(K, V))
beta = beta / beta.sum(axis=1, keepdims=True)

def norm(arr):
    return np.sqrt(np.sum(np.square(arr)))

def step_EM(phi, gamma, alpha, beta, w, max_step=100):
    # w: M x N
    assert w.max() + 1 == V
    w_reshape = w.reshape(-1)
    print(w.shape)
    print(w_reshape.size)
    print(w.max() + 1)
    w_ind = np.zeros((w_reshape.size, w_reshape.max()+1))
    w_ind[np.arange(w_reshape.size), w_reshape] = 1
    w_ind = w_ind.reshape(M, N, V)
    criteria_M, criteria_E = 1., 1.
    # i = 0
    t0 = time.time()
    for i in range(max_step):
        # record
        # E-step
        count_E, count_M = 0, 0
        while criteria_E > 1e-2 or count_E == 0:
            phi_old = phi
            gamma_old = gamma
            log_theta_exp = digamma(gamma) - digamma(np.sum(gamma, axis=1, keepdims=True))

            beta_iw = beta[:, w].transpose(1, 2, 0) # (K, M, N) -> (M, N, K)
            exp_theta = np.tile(np.expand_dims(np.exp(log_theta_exp), 1), (1, N, 1)) # (M, K) -> (M, N, K)
            phi = beta_iw * exp_theta # (M, N, K)
            phi /= np.sum(phi, axis=-1, keepdims=True)

            gamma = np.tile(np.expand_dims(alpha, 0), (M, 1)) + np.sum(phi, axis=1)
            # print(gamma[-1])
            criteria_E = 1 / float(2*M) * (norm(phi - phi_old) + norm(gamma - gamma_old))
            count_E += 1
            # print('E step ', count_E, ':', criteria_E)
        # M-step
        while criteria_M > 1e-4 or count_M == 0:
            alpha_old = alpha
            beta = np.einsum("mnk, mnv -> kv", phi, w_ind)
            beta = beta / beta.sum(axis=1, keepdims=True)
            
            g = M * (digamma(np.sum(alpha, axis=0, keepdims=True)) - digamma(alpha)) + np.sum(digamma(gamma) - digamma(np.sum(gamma, axis=1, keepdims=True)), axis=0) # (K, )
            h = M * polygamma(1, alpha)
            # z = -polygamma(1, np.sum(alpha, axis=0, keepdims=True))
            z = -polygamma(1, np.sum(alpha))
            # c = np.sum(g, keepdims=True) / h / (1.0/z + np.sum(1.0/h, keepdims=True))
            c = np.sum(g/h) / (1.0/z + np.sum(1.0/h))
            alpha = alpha + (g - c) / h

            criteria_M = norm(alpha - alpha_old)
            count_M += 1
            # print('M step ', count_M, ':', criteria_M)
        
        # i += 1
        print("Epoch", i+1,  ":", criteria_E, criteria_M, '; Time: ', time.time()-t0, '; E-step:', count_E, '; M-step:', count_M)
        # print('=====================================')
    return alpha, beta, gamma

if __name__ == "__main__":
    exp_name = "eco_100"
    max_step = 100
    w = combine_preprocess()
    alpha, beta, gamma = step_EM(phi, gamma, alpha, beta, w, max_step)
    np.save("alpha_"+exp_name+".npy", alpha)
    np.save("beta_"+exp_name+".npy", beta)
    np.save("gamma_"+exp_name+".npy", gamma)

    # https://thehill.com/regulation/court-battles/600020-judge-rules-trumps-efforts-to-overturn-election-likely-criminal
    # https://www.inquirer.com/sixers/76ers-suns-joel-embiid-devin-booker-first-east-20220327.html



