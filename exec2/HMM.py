from math import log

n_states = 2
prob_init = [.5, .5]
a_trans   = [[.8, .2],
             [.2, .8]]

b_emiss = [{'A' : 0.4, 'C' : 0.1, 'G' : 0.4, 'T' : 0.1},
           {'A' : 0.1, 'C' : 0.4, 'G' : 0.1, 'T' : 0.4}]


def backward(O):
    T = len(O)
    b = [[0 for _ in range(n_states)] for _ in range(T)]
    b[T-1] = [1 for _ in range(n_states)]

    for t in range(T-2, -1, -1):
        for i in range(n_states):
            for j in range(n_states):
                b[t][i] += a_trans[i][j] * b_emiss[j][O[t+1]] * b[t+1][j]
    p = .0
    for j in range(n_states):
        p += prob_init[j] * b_emiss[j][O[0]] * b[0][j]

    return p, b


def forward(O):
    T = len(O)
    a = [[0 for _ in range(n_states)] for _ in range(T)]
    a[0] = [prob_init[i] * b_emiss[i][O[0]] for i in range(n_states)]

    for t in range(1, T):
        for j in range(n_states):
            for i in range(n_states):
                a[t][j] += a[t-1][i] * a_trans[i][j] * b_emiss[j][O[t]]


    p = sum(a[T-1])

    return p, a


def viterbi(O):
    T = len(O)
    back = [[-1 for _ in range(n_states)] for _ in range(T)]
    v = [[0 for _ in range(n_states)] for _ in range(T)]
    v[0] = [prob_init[i] * b_emiss[i][O[0]] for i in range(n_states)]

    for t in range(1,T):
        for j in range(n_states):
            for i in range(n_states):
                new_term = v[t-1][i] * a_trans[i][j] * b_emiss[j][O[t]]
                if v[t][j] < new_term:
                    v[t][j] = new_term
                    back[t][j] = i

    score = max(v[T-1])
    print('Max score: {}'.format(log(score)))
    print('---------')

    q = v[T-1].index(score)
    for t in range(T-1, -1, -1):
        print("t = {}, state = {}".format(t+1, q))
        q = back[t][q]

    return log(score)


def posterior(O):

    T = len(O)
    p, a = forward(O)
    _, b = backward(O)

    gamma = [[a[t][j] * b[t][j] / p for j in range(n_states)] for t in range(T)]

    return gamma


if __name__ == "__main__":
    O = 'CGTCAG'
    T = len(O)

    p, a = forward(O)

    print("logarithm of P(O | hmm_params) = {}".format(log(p)))
    print(' --------- ')
    for t in range(T):
        print("Forward matrix at time t = {}".format(t+1))
        for i in range(n_states):
            print("  State S{}:  {:.5} ".format(i+1, a[t][i]))


    print('\n \n')
    gamma = posterior(O)
    for t in range(T):
        print("Posterior Probability at time t = {} being in state S1: {:.2}".format(t+1, gamma[t][0]))

    print('\n \n')
    _ = viterbi(O)
