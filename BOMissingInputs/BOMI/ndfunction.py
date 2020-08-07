import numpy as np

from BOMI.Space import Space

class SyntheticFunction():
    def __init__(self, dim):
        self.numCalls = 0
        self.input_dim = dim
        self.nbounds = [(0, 1)] * self.input_dim
    def plot(self):
        print ("not implemented")

    def randInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = np.random.uniform(i[0], i[1])
            rand.append(tmpRand)
        return rand

    def randUniformInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = np.random.uniform(i[0], i[1])
            rand.append(tmpRand)
        return rand

    def randUniformInNBounds(self):
        rand = []
        for i in range(0,self.input_dim):
            tmpRand = np.random.uniform(self.nbounds[i][0], self.nbounds[i][1])
            rand.append(tmpRand)
        return rand

    def normalize(self,x):
        val = []
        for i in range(0,self.input_dim):
            val.append((x[i] - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0]))
        return val

    def denormalize(self,x):
        val = []
        for i in range(0, self.input_dim):
            val.append(1.0 * (x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]))
        return val

class Eggholder(SyntheticFunction):
    def __init__(self, dim, missingRate, missingNoise, isRandomMissing=False):
        SyntheticFunction.__init__(self, 2)  # Constructor of the parent class
        self.input_dim = 2
        self.numCalls = 0
        #self.bounds = OrderedDict({'x1': (-5, 10), 'x2': (0, 15)})
        self.bounds = [(-512, 512), (-512, 512)]
        self.nbounds = [(0,1),(0,1)]
        self.min = [512, 404.2319]
        self.fmin = -959.6407
        self.ismax = -1
        self.name = 'Eggholder'
        self.discreteIdx = []
        self.categoricalIdx = []

        self.missRate = missingRate
        self.missNoise = missingNoise
        self.isRandomMissing = isRandomMissing

    def func(self,X):
        self.numCalls+=1
        X=np.asarray(X)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]

        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        fx = term1 + term2
        return fx * self.ismax

    def func_with_missing(self,X):
        self.numCalls+=1
        X=np.asarray(X)

        # For missing rate
        missingX = X.copy()
        actualX = X.copy()
        for i in range(self.input_dim):
            prob = np.random.uniform(0, 1, 1)
            if prob < self.missRate:
                missingX[i] = np.nan
                if self.isRandomMissing:
                    actualX = self.randInBounds()
                else:
                    # actualX[i] = X[i] + np.random.normal(0,1,1)
                    actualX[i] = X[i] + np.random.uniform(-np.abs(self.bounds[i][1] - self.bounds[i][0])*self.missNoise, np.abs(self.bounds[i][1] - self.bounds[i][0])*self.missNoise, 1)
                    # actualX[i] = X[i] + np.random.uniform(-X[i] * self.missNoise, X[i] * self.missNoise, 1)
                    actualX[i] = np.clip(actualX[i], a_min=self.bounds[i][0], a_max=self.bounds[i][1])
                break
            # else:
            #     actualX[i] = X[i] + np.random.normal(0, 1, 1)

        if len(X.shape)==1:
            x1=actualX[0]
            x2=actualX[1]
        else:
            x1=actualX[:,0]
            x2=actualX[:,1]

        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        fx = term1 + term2
        if not np.isscalar(fx):
            fx = fx[0]
        return fx * self.ismax, missingX

class Schubert_Nd(SyntheticFunction):
    # Min −186.7309 at 18 global minima
    def __init__(self, dim, missingRate, missingNoise, isRandomMissing):
        SyntheticFunction.__init__(self, dim) # Constructor of the parent class
        self.numCalls = 0
        self.input_dim = dim
        self.bounds = [(-10, 10)] * self.input_dim
        self.nbounds = [(0, 1)] * self.input_dim
        self.name = 'SchubertNd'
        self.fmin = 186.7309
        #ismax is 1 since we already flip -1 in each function
        self.ismax = -1
        self.discreteIdx = []
        self.categoricalIdx = []
        self.spaces = []
        for i in range(0,dim):
            self.spaces.append(Space(-10, 10, None))
            self.spaces[i].setValuesSet([i for i in range(-10,11)])

        self.missRate = missingRate
        self.missNoise = missingNoise
        self.isRandomMissing = isRandomMissing

    def _interfunc(self,X):
        X = np.asarray(X)
        fx = 1
        for i in X:
            s = 0
            for j in range(1,6):
                s = s + j * np.cos((j + 1) * i + j)
            fx *= s
        return fx * self.ismax


    def func(self,X):
        self.numCalls += 1
        return self._interfunc(X)

    def func_with_missing(self, X):
        # For missing rate
        missingX = X.copy()
        actualX = X.copy()
        for i in range(self.input_dim):
            prob = np.random.uniform(0, 1, 1)
            if prob < self.missRate:
                missingX[i] = np.nan
                if self.isRandomMissing:
                    actualX = self.randInBounds()
                else:
                    # actualX[i] = X[i] + np.random.normal(0, 1, 1)
                    # actualX[i] = X[i] + np.random.normal(0.0, np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
                    actualX[i] = X[i] + np.random.uniform(
                        -np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise,
                        np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
                    clipTmp = np.clip(actualX[i], a_min=self.bounds[i][0], a_max=self.bounds[i][1])
                    if(np.isscalar(clipTmp)):
                        actualX[i] = clipTmp
                    else:
                        actualX[i] = clipTmp[0]
                break
        X = actualX
        fx = self._interfunc(X)
        if not np.isscalar(fx):
            fx = fx[0]
        return fx, missingX

class Alpine_Nd(SyntheticFunction):
    # Min 0 at [0, ..., 0]

    def __init__(self, dim, missingRate, missingNoise, isRandomMissing):
        SyntheticFunction.__init__(self, dim)
        self.numCalls = 0
        self.input_dim = dim
        self.bounds = [(-10, 10)] * self.input_dim
        self.nbounds = [(0, 1)] * self.input_dim
        self.name = 'AlpineNd'
        self.fmin = 0.0
        #ismax is 1 since we already flip -1 in each function
        self.ismax = -1
        self.discreteIdx = []
        self.categoricalIdx = []
        self.spaces = []
        for i in range(0,dim):
            self.spaces.append(Space(-10, 10, None))
            self.spaces[i].setValuesSet([i for i in range(-10,11)])

        self.missRate = missingRate
        self.missNoise = missingNoise
        self.isRandomMissing = isRandomMissing


    def _interfunc(self,X):
        X=np.asarray(X)
        fx = 0
        for i in range(0,self.input_dim):
            fx += abs(X[i] * np.sin(X[i]) + 0.1 * X[i])
        return fx * self.ismax

    def func(self,X):
        self.numCalls += 1
        return self._interfunc(X)

    def func_with_missing(self,X):
        self.numCalls+=1
        # For missing rate
        # missingX = X.copy()
        # actualX = X.copy()
        # for i in range(self.input_dim):
        #     prob = np.random.uniform(0, 1, 1)
        #     if prob < self.missRate:
        #         missingX[i] = np.nan
        #         if self.isRandomMissing:
        #             actualX = self.randInBounds()
        #         else:
        #             # actualX[i] = X[i] + np.random.normal(0.0,np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
        #             actualX[i] = X[i] + np.random.uniform(
        #                 -np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise,
        #                 np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
        #             actualX[i] = np.clip(actualX[i], a_min=self.bounds[i][0], a_max=self.bounds[i][1])
        #         break
        missingX = X.copy()
        actualX = X.copy()
        countMiss = 0
        for i in range(self.input_dim):
            prob = np.random.uniform(0, 1, 1)
            if prob < self.missRate:
                missingX[i] = np.nan
                if self.isRandomMissing:
                    actualX = self.randInBounds()
                else:
                    # actualX[i] = X[i] + np.random.normal(0, 0.05, 1)
                    # actualX[i] = X[i] + np.random.normal(0.0, 1.0, 1)
                    # actualX[i] = X[i] + np.random.normal(0.0, np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
                    # actualX[i] = X[i] + np.random.uniform(
                    #     -np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise,
                    #     np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)

                    prob2 = np.random.uniform(0, 1, 1)
                    if prob2 < 0.5:
                        actualX[i] = X[i] + np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise
                    else:
                        actualX[i] = X[i] - np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise
                    actualX[i] = np.clip(actualX[i], a_min=self.bounds[i][0], a_max=self.bounds[i][1])
                    countMiss += 1
                if countMiss == 4:
                    break
        X = actualX
        fx = self._interfunc(X)
        if not np.isscalar(fx):
            fx = fx[0]
        return fx, missingX

class Schwefel_Nd(SyntheticFunction):
    def __init__(self, dim, missingRate, missingNoise, isRandomMissing):
        SyntheticFunction.__init__(self, dim)
        self.name = 'SchwefelNd'
        self.numCalls = 0
        self.input_dim = dim
        self.bounds = [(-500.0, 500.0)] * self.input_dim
        self.nbounds = [(0, 1)] * self.input_dim
        self.fmin = 0.0
        # ismax is 1 since we already flip -1 in each function
        self.ismax = -1

        self.missRate = missingRate
        self.missNoise = missingNoise
        self.isRandomMissing = isRandomMissing

    def func(self,X):
        self.numCalls+=1
        X=np.asarray(X)

        d = len(X)
        sum = 0
        for ii in range(d):
            xi = X[ii]
            sum = sum + xi * np.sin(np.sqrt(abs(xi)))
        fx = 418.9829 * d - sum

        return fx * self.ismax

    def func_with_missing(self,X):
        self.numCalls+=1
        X=np.asarray(X)

        # For missing rate
        missingX = X.copy()
        actualX = X.copy()
        for i in range(self.input_dim):
            prob = np.random.uniform(0, 1, 1)
            if prob < self.missRate:
                missingX[i] = np.nan
                if self.isRandomMissing:
                    actualX = self.randInBounds()
                else:
                    # actualX[i] = X[i] + np.random.normal(0, 50, 1)
                    # actualX[i] = X[i] + np.random.normal(0.0, np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
                    actualX[i] = X[i] + np.random.uniform(
                        -np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise,
                        np.abs(self.bounds[i][1] - self.bounds[i][0]) * self.missNoise, 1)
                    actualX[i] = np.clip(actualX[i], a_min=self.bounds[i][0], a_max=self.bounds[i][1])
                break
            # else:
            #     actualX[i] = X[i] + np.random.normal(0, 1, 1)

        X = actualX

        d = len(actualX)
        sum = 0
        for ii in range(d):
            xi = X[ii]
            sum = sum + xi * np.sin(np.sqrt(abs(xi)))
        fx = 418.9829 * d - sum
        return fx * self.ismax, missingX

