import numpy as np
from itertools import permutations
import math

#--- Hamiltonian path

class HamiltonianDecoder():
    """
    Decode Reading order as the best hamiltonian path
    """
    def __init__(self, P):
        self._P = P
        self.N = self._P.shape[0]
        self.id = 'hamiltonian'

    def _brute_force(self):
        """
        use brute force to get the best path
        """
        if self.N > 9:
            #print("Input set is too big for brute force estimation.")
            self.best_path = None
        else:
            #print("Number of permutations to check: {}".format(math.factorial(self.N)))
            #init = 
            A = self._P + np.finfo(np.float).eps
            A = (A + (1-A).T)/2
            for i in range(A.shape[0]):
                A[i,i] = np.finfo(np.float).eps
            init = (A>0.5).sum(axis=1).argsort()[::-1]
            #--- use log(p(Y=1\mid s',s)) to shift multiplication to sum
            lP = np.log(A)
            for i in range(lP.shape[0]):
                lP[i,i] = 0
            #init_cost = 0
            ##--- lP[x:x+1] está MAL hay que sumar respecto a i+1 en z, no en lP.
            #for i in range(len(init)-1):
            #    init_cost += lP[init[i],init[i+1]:].sum()
            z_star = []
            z_cost = -np.inf
            for z in permutations(range(self.N)):
                cost = 0
                for i in range(len(z)-1):
                    cost += lP[z[i],z[i+1:]].sum()
                if cost > z_cost:
                    z_cost = cost
                    z_star = z
            self.best_path = np.array(z_star)

    def _dinamic_decode(self):
        """
        use dinamic programing to reduce the number of sums to be realized
        """
        raise NotImplementedError
    def _random(self, P):
        return random.sample(list(range(self.N)), self.N)

    def _branch_and_bound(self, **kwargs):
        if self.N > 100:
            print("Input set is too big for brute force estimation.")
            self.best_path = None
        else:
            A = self._P + np.finfo(np.float).eps
            A = (A + (1-A).T)/2
            for i in range(A.shape[0]):
                A[i,i] = np.finfo(np.float).eps
            #--- use log(p(R\mid s',s)) to shift multiplication to sum
            lP = np.log(A)
            for i in range(lP.shape[0]):
                lP[i,i] = 0
            self.best_path = branch_and_bound(lP, kwargs)
    def run(self):
        self._brute_force()

class branch_and_bound():
    """
    """
    def __init__(self, P, **kwargs):
        self._P = P
        init_func = kwargs["init"]
        objective_func = kwargs["objective"]
        bounding_func = kwargs["bounding"]


#---- Voting Decoder

class VotingDecoder():
    """
    A simple decoder, it just count the number of 0's in the normalized matrix 
    """
    def __init__(self, P):
        self._P = P
        self.id = 'voting'
    def run(self):
        A = self._P + np.finfo(np.float).eps
        A = A/(A+A.T)
        for i in range(A.shape[0]):
            A[i,i] = np.finfo(np.float).eps
        T = (A>0.5).sum(axis=1)
        self.best_path = T.argsort()[::-1]

class CountDecoder():
    """
    A simple decoder, it just count the number of 0's in the simetrical matrix
    """
    def __init__(self, P):
        self._P = P
        self.id = 'count'
    def run(self):
        A = self._P + np.finfo(np.float).eps
        A = (A + (1-A).T)/2
        for i in range(A.shape[0]):
            A[i,i] = np.finfo(np.float).eps
        T = (A>0.5).sum(axis=1)
        self.best_path = T.argsort()[::-1]


#---- Greedy Decoder

class GreedyDecoder():
    """
    A greedy decoder of order-relation matrix. For each position in the
    reading order we select the most probable one, then move to the next 
    position. Most probable for position:
    z^{\star}_t = \argmax_{(s,\nu) \ni z^{\star}}
        \prod_{(s',\nu') \in z^\star}{\tilde{P}(Y=1\mid s',s)}
        \times \prod_{\substack{(s'',\nu'') \ni z^\star\\
         s'' \ne s}}{\tilde{P}(r=0\mid s'',s)}, 1\le t \le n
    """
    def __init__(self, P):
        self._P = P
        self.N = self._P.shape[0]
        self.id = 'greedy'

    def run(self):
        A = self._P + np.finfo(np.float).eps
        A = (A + (1-A).T)/2
        for i in range(A.shape[0]):
            A[i,i] = np.finfo(np.float).eps
        self.best_path = []
        #--- use log(p(R\mid s',s)) to shift multiplication to sum
        lP = np.log(A)
        for i in range(self.N):
            lP[i,i] = 0
        for t in range(self.N):
            #print(lP)
            #print("----------------------")
            for i in range(self.N):
                idx = np.argmax(lP.sum(axis=1))
                if idx not in self.best_path:
                    self.best_path.append(idx)
                    lP[idx,:] = lP[:,idx]
                    lP[:,idx] = 0
                    break
        self.best_path = np.array(self.best_path)
        #self.best_path = np.argsort(CumProb)[::-1]

#---- Forward-Backward Decoder

class ForwardBackwardDecoder():
    """
    Class to define Forward-Backward decoder, implemented step by step 
    in order to take full control of changes if necesary (until full 
    theory is revised)
    """
    def __init__(self, P, init=None, end=None):
        self._P = P
        self.N = self._P.shape[0]
        self.id = 'fwd-bwd'
        if init is not None:
            self._s0 = init
        else:
            self._s0 = np.ones(self.N)
        if end is not None:
            self._sN = end
        else:
            self._sN = np.ones(self.N)

    def _forward(self):
        self.alpha = np.zeros((self.N, self.N))
        self.alpha[0,:] = self._s0
        PT = self._P.T
        for t in range(1,self.N):
            self.alpha[t,:] = (self.alpha[t-1,:]*PT).sum(axis=1)

    def _denominator(self):
        self.Z = (self.alpha[-1,:]*self._sN).sum()

    def _backward(self):
        self.beta = np.zeros((self.N, self.N))
        self.beta[-1,:] = self._sN
        for t in range(self.N-2, -1, -1):
            self.beta[t,:] = (self.beta[t+1,:]*self._P).sum(axis=1)
        
        #Z = self.beta.sum(axis=1)
        #self.beta = self.beta / Z[:,None]

    def _compute_probs(self):
        self.prob = (self.alpha * self.beta)/self.Z

    def _decode_by_step(self):
        self.best_path = np.argmax(self.prob, axis=1) 

    def _decode_by_order(self):
        """
        Decode the best path as $\pi_t = \argmax_{s \in S} p(\nu_s = t \mid S)
        without repetitions in order from 0 to N.
        """
        self.best_path = []
        S = np.argsort(-self.prob)
        for i in range(self.N):
            for idx in S[i]:
                if idx not in self.best_path:
                    self.best_path.append(idx)
                    break 
        self.best_path = np.array(self.best_path, dtype = np.int)
        if len(self.best_path) == self.N:
            return True
        else:
            return False
    
    def _decode_by_maxprob(self):
        """
        Decode the best path as $\pi_t = \argmax_{s \in S} p(\nu_s = t \mid S)
        without repetitions in order from the most probable to the lesser.
        """
        self.best_path = np.zeros(self.N, dtype=np.int)
        P = self.prob.copy()
        
        for i in range(self.N):
            idx = np.unravel_index(np.argmax(P, axis=None), P.shape)
            self.best_path[idx[0]] = idx[1]
            P[idx[0],:] = 0
            P[:,idx[1]] = 0


    def run(self):
        self._P = self._P + np.finfo(np.float).eps
        self._P = (self._P + (1-self._P).T)/2
        for i in range(self._P.shape[0]):
            self._P[i,i] = np.finfo(np.float).eps
        self._forward()
        self._backward()
        self._denominator()
        self._compute_probs()
        self._decode_by_maxprob()

#--- 
#DECODERS=[VotingDecoder, CountDecoder, GreedyDecoder, ForwardBackwardDecoder, HamiltonianDecoder]
#DECODERS=[CountDecoder, GreedyDecoder, HamiltonianDecoder]
DECODERS=[CountDecoder, GreedyDecoder]

def test():
    P = np.array([[0.0, 0.9, 0.9, 0.9],
                  [0.1, 0.0, 0.9, 0.9],
                  [0.1, 0.1, 0.0, 0.9],
                  [0.1, 0.1, 0.1, 0.0]])
    decoder = HamiltonianDecoder(P)
    decoder.run()
    print(decoder.best_path)

if __name__=="__main__":
    test()
