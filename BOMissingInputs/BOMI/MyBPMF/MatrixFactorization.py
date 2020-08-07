import numpy as np
from numpy.linalg import norm
from numpy.random import multivariate_normal
from scipy.stats import wishart

from joblib import Parallel, delayed
import multiprocessing

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != -1)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] != -1
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # if (i + 1) % 10 == 0:
            #     print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)

class PMF(object):
    '''
    Probabilistic Matrix Factorization
    '''

    def __init__(self, n_feature, epsilon, lam, n_epoches, n_batches):
        self.n_feature = n_feature  # number of features
        self.epsilon = epsilon  # epsilon for leanring rate
        self.lam = lam  # lambda for L2 regularization

        self.n_epoches = n_epoches  # number of epoches
        self.n_batches = n_batches  # number of batches

        self.V = None  # items feature matrix
        self.U = None  # users feature matrix

    def loss(self, ratings):
        '''
        Loss Function for evaluating matrix U and V
        '''
        errors = [
            (float(r_ij) - np.dot(self.U[int(i)], self.V[int(j)].T)) ** 2 + \
            self.lam * norm(self.U[int(i)]) + self.lam * norm(self.V[int(j)])
            for i, j, r_ij in ratings]

        # for i in range(len(ratings)):
        #     for j in range(len(ratings[i])):
        #         r_ij = ratings[i][j]
        #         errors = [
        #             (float(r_ij) - np.dot(self.U[i], self.V[j].T)) ** 2 + \
        #             self.lam * norm(self.U[i]) + self.lam * norm(self.V[j])]
        return sum(errors)

    def sgd_update(self, ratings):
        '''
        Update matrix U and V by Stochastic Gradient Descent.
        '''
        # print(self.U)
        # print(self.V.shape)
        # for i in range(len(ratings)):
        #     for j in range(len(ratings[i])):
        #         r_ij = ratings[i][j]
        #         if r_ij != -1:
        #             r_ij_hat = np.dot(self.U[i], self.V[j].T)
        #             grad_U_i = (r_ij_hat - float(r_ij)) * self.V[j] + self.lam * self.U[i]
        #             grad_V_j = (r_ij_hat - float(r_ij)) * self.U[i] + self.lam * self.V[j]
        #             self.U[i] = self.U[i] - self.epsilon * grad_U_i
        #             self.V[j] = self.V[j] - self.epsilon * grad_V_j

        for i, j, r_ij in ratings:
            i = int(i)
            j = int(j)
            r_ij_hat = np.dot(self.U[i], self.V[j].T)
            grad_U_i = (r_ij_hat - float(r_ij)) * self.V[j] + self.lam * self.U[i]
            grad_V_j = (r_ij_hat - float(r_ij)) * self.U[i] + self.lam * self.V[j]
            self.U[i] = self.U[i] - self.epsilon * grad_U_i
            self.V[j] = self.V[j] - self.epsilon * grad_V_j

    def sgd(self,ratings):
        """
        Perform stochastic graident descent
        """
        for i, j, r in ratings:
            i = int(i)
            j = int(j)
            # Computer prediction and error
            r_ij_hat = np.dot(self.U[i], self.V[j].T)
            e = (r - r_ij_hat)

            # Update biases
            self.b_u[i] += self.epsilon * (e - self.lam * self.b_u[i])
            self.b_i[j] += self.epsilon * (e - self.lam * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.U[i, :][:]

            # Update user and item latent feature matrices
            self.U[i, :] += self.epsilon * (e * self.U[j, :] - self.lam * self.U[i, :])
            self.V[j, :] += self.epsilon * (e * P_i - self.lam * self.V[j, :])


    def fit(self, train_ratings, test_ratings):
        '''
        Fit PMF model with respect to the ratings. A rating is a triple (user,
        item, rating), in particular, user and item are integers to indicate
        unique ids respectively, and rating is a real value score that associates
        with corresponding user and item. For here, ratings is a numpy array
        with shape (n, 3).
        Params:
        - train_ratings: ratings entries for training purpose
        - test_ratings:  ratings entries for testing purpose
        '''
        # get number of training samples and testing samples
        # n_trains = train_ratings.shape[0]
        # n_tests = test_ratings.shape[0]
        # # get number of items and number of users
        # n_users = int(max(np.amax(train_ratings[:, 0]), np.amax(test_ratings[:, 0]))) + 1
        # n_items = int(max(np.amax(train_ratings[:, 1]), np.amax(test_ratings[:, 1]))) + 1
        n_trains = train_ratings.shape[0]
        n_tests = test_ratings.shape[0]
        # get number of items and number of users
        n_users = len(train_ratings)
        n_items = len(train_ratings[0])

        train_ratings = np.array([
            (i, j, train_ratings[i, j])
            for i in range(n_users)
            for j in range(n_items)
            if train_ratings[i, j] != -1
        ])

        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.b = np.mean(train_ratings[np.where(train_ratings != -1)])

        # Initialization
        if self.V is None or self.U is None:
            self.e = 0
            # self.U = 0.1 * np.random.randn(n_users, self.n_feature)
            # self.V = 0.1 * np.random.randn(n_items, self.n_feature)
            self.U = np.random.normal(scale=1. / self.n_feature, size=(n_users, self.n_feature))
            self.V = np.random.normal(scale=1. / self.n_feature, size=(n_items, self.n_feature))

        loss_arr = []
        # training iterations over epoches
        while self.e < self.n_epoches:
            self.e += 1
            # shuffle training samples
            shuffled_order = np.arange(n_trains)
            np.random.shuffle(shuffled_order)
            # training iterations over batches
            avg_train_loss = []
            avg_test_loss = []
            batch_size = int(n_trains / self.n_batches)
            for batch in range(self.n_batches):
                idx = np.arange(batch_size * batch, batch_size * (batch + 1))
                batch_idx = np.mod(idx, n_trains).astype('int32')
                # training ratings selected in current batch
                batch_ratings = train_ratings[shuffled_order[batch_idx], :]
                # test ratings sample with the same size as the training batch
                # sample_test_ratings = test_ratings[np.random.choice(n_tests, batch_size), :]
                # update U and V by sgd in a close-formed gradient
                # print(batch_ratings)
                self.sgd_update(batch_ratings)
                # self.sgd(batch_ratings)
                # print("U:",self.U)
                # print("V:", self.V)
                # print("R:", np.dot(self.U.T, self.V))
                # loss for training and testing U, V and ratings
                train_loss = self.loss(batch_ratings)
                # test_loss = self.loss(sample_test_ratings)
                avg_train_loss.append(train_loss)
                # avg_test_loss.append(test_loss)
            # training log ouput
            avg_train_loss = np.mean(avg_train_loss) / float(batch_size)
            # avg_test_loss = np.mean(avg_test_loss) / float(batch_size)
            # print('Training loss:\t%f' % avg_train_loss, file=sys.stderr)
            # print('Testing loss:\t%f' % avg_test_loss, file=sys.stderr)
            loss_arr.append(avg_train_loss)

def Normal_Wishart(mu_0, lamb, W, nu, seed=None):
    """Function extracting a Normal_Wishart random variable"""
    # first draw a Wishart distribution:
    Lambda = wishart(df=nu, scale=W, seed=seed).rvs()  # NB: Lambda is a matrix.
    # then draw a Gaussian multivariate RV with mean mu_0 and(lambda*Lambda)^{-1} as covariance matrix.
    cov = np.linalg.inv(lamb * Lambda)  # this is the bottleneck!!
    mu = multivariate_normal(mu_0, cov)
    return mu, Lambda, cov

def reduce_matrix(N_max, M_max, filename, correspondence_list_users, correspondence_list_movies, sep=" "):
    """In some datasets, the movies and users have a certain identifier that corresponds to one
       of a larger dataset; this means not all the user/movie identifier are used. Then it is better to
       reduce the matrix, in order to have a smaller representation.  We assume to have a correspondence list
       both for users and movies, i.e. a list where element i indicates the i-th used identifier; ex:
       correspondence_list_users = [1,3,7] means that the users 1,3,7 are respectively the 1st, 2nd and 3rd. Then
       they could be renamed in this way, saving a lot of space.
       """

    # first call ranking_matrix on the filename, generating a big matrix (many rows/columns will be empty)
    R = ranking_matrix(N_max, M_max, filename, sep)

    N_actual = len(correspondence_list_users)
    M_actual = len(correspondence_list_movies)

    R_reduced = np.zeros((N_actual, M_actual))

    for i, user in enumerate(correspondence_list_users):
        for j, movie in enumerate(correspondence_list_movies):
            R_reduced[i, j] = R[correspondence_list_users[i] - 1, correspondence_list_movies[j] - 1]

    return R_reduced

def ranking_matrix(N, M, filename, sep=" "):
    """Function creating the NxM rating matrix from filename.
    It assumes that the file contains on every line a triple (user, movie, ranking).
    Moreover, users' and movies are numbered starting from 1.
    """
    R = np.zeros((N, M))
    f = open(filename, "r")
    for line in f:
        if line[0] == '%':
            # this is a comment
            continue
        (user, movie, ranking) = line.split(sep)
        R[np.int(user) - 1, np.int(movie) - 1] = np.int(ranking)
    return R

def read_correspondence_list(filename):
    """Function reading the correspondence list from a -mtx file "filename"""
    corr_list = []
    f = open(filename, "r")
    for line in f:
        if line[0] == '%':
            continue
        corr_list.append(np.int(line))
    return corr_list

class myBPMF():
    def __init__(self):
        # posterior of next X and Y
        self.method = "BPMF"

    def ParallelBPMF(self, R, R_test, U_in, V_in, T, D, initial_cutoff, lowest_rating, highest_rating, in_alpha, output_file,
             mu_0=None, Beta_0=None, W_0=None, nu_0=None, missing_mask = -1, save_file=True):
        """
        R is the ranking matrix (NxM, N=#users, M=#movies); we are assuming that R[i,j]=0 means that user i has not ranked movie j
        R_test is the ranking matrix that contains test values. Same assumption as above.
        U_in, V_in are the initial values for the MCMC procedure.
        T is the number of steps.
        D is the number of hidden features that are assumed in the model.

        mu_0 is the average vector used in sampling the multivariate normal variable
        Beta_0 is a coefficient (?)
        W_0 is the DxD scale matrix in the Wishart sampling
        nu_0 is the number of degrees of freedom used in the Wishart sampling.

        U matrices are DxN, while V matrices are DxM.

        If save_file=True, this function internally saves the file at each iteration; this results in a different file for each value
        of D and is useful when the algorithm may stop during the execution.
        """

        # @njit
        def ranked(i, j):  # function telling if user i ranked movie j in the train dataset.
            if R[i, j] != missing_mask:
                return True
            else:
                return False

        # @njit
        def ranked_test(i, j):  # function telling if user i ranked movie j in the test dataset.
            if R_test[i, j] != missing_mask:
                return True
            else:
                return False

        N = R.shape[0]
        M = R.shape[1]

        R_predict = np.zeros((N, M))
        U_old = np.array(U_in)
        V_old = np.array(V_in)

        train_err_list = []
        test_err_list = []
        train_epoch_list = []

        # initialize now the hierarchical priors:
        alpha = in_alpha  # observation noise, they put it = 2 in the paper
        # mu_u = np.zeros((D, 1))
        # mu_v = np.zeros((D, 1))
        # Lambda_U = np.eye(D)
        # Lambda_V = np.eye(D)

        # COUNT HOW MAY PAIRS ARE IN THE TEST AND TRAIN SET:
        # pairs_test = 0
        # pairs_train = 0
        # for i in range(N):
        #     for j in range(M):
        #         if ranked(i, j):
        #             pairs_train = pairs_train + 1
        #         if ranked_test(i, j):
        #             pairs_test = pairs_test + 1
        #
        # print(pairs_test, pairs_train)

        # SET THE DEFAULT VALUES for Wishart distribution
        # we assume that parameters for both U and V are the same.

        if mu_0 is None:
            mu_0 = np.zeros(D)
        if nu_0 is None:
            nu_0 = D
        if Beta_0 is None:
            Beta_0 = 2
        if W_0 is None:
            W_0 = np.eye(D)

        # results = pd.DataFrame(columns=['step', 'train_err', 'test_err'])

        # parameters common to both distributions:
        Beta_0_star = Beta_0 + N
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)  # compute the inverse once and for all

        def SampleUser(i, inV, inMu_U, inLambda_U):
            Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for j in range(M):  # loop over the movies
                if ranked(i, j):  # only if movie j has been ranked by user i!
                    Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(inV[:, j], ndmin=2)),
                                                     np.array((inV[:, j]), ndmin=2))  # CHECK
                    # Lambda_U_2 = Lambda_U_2 + npdot(nptranspose(np.array(inV[:, j], ndmin=2)),
                    #                                  np.array((inV[:, j]), ndmin=2))  # CHECK
                    mu_i_star_1 = inV[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_i_star_U = inLambda_U + alpha * Lambda_U_2
            Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

            ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
            mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_U, inMu_U)
            mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
            output = multivariate_normal(mu_i_star, Lambda_i_star_U_inv)

            return output

        def SampleFeature(input_U_new, j, inMu_V, inLambda_V):
            Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for i in range(N):  # loop over the movies
                if ranked(i, j):
                    Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(input_U_new[:, i], ndmin=2)),
                                                     np.array((input_U_new[:, i]), ndmin=2))
                    # Lambda_V_2 = Lambda_V_2 + npdot(np.transpose(np.array(input_U_new[:, i], ndmin=2)),
                    #                                  np.array((input_U_new[:, i]), ndmin=2))
                    mu_i_star_1 = input_U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_j_star_V = inLambda_V + alpha * Lambda_V_2
            Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)
            # Lambda_j_star_V_inv = nplinalginv(Lambda_j_star_V)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_V, inMu_V)
            mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            # mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_V, inMu_V)
            # mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            output = multivariate_normal(mu_j_star, Lambda_j_star_V_inv)

            # return output, mu_j_star, Lambda_j_star_V_inv
            return output

        for t in range(T):
            # print("Step ", t)
            # FIRST SAMPLE THE HYPERPARAMETERS, conditioned on the present step user and movie feature matrices U_t and V_t:

            # movie hyperparameters:
            V_average = np.sum(V_old, axis=1) / N  # in this way it is a 1d array!!
            S_bar_V = np.dot(V_old, np.transpose(V_old)) / N
            # S_bar_V = npdot(V_old, np.transpose(V_old)) / N

            mu_0_star_V = (Beta_0 * mu_0 + N * V_average) / (Beta_0 + N)
            W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
            # W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * npdot(
            #     np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
            W_0_star_V = np.linalg.inv(W_0_star_V_inv)
            # W_0_star_V = nplinalginv(W_0_star_V_inv)
            mu_V, Lambda_V, cov_V = Normal_Wishart(mu_0_star_V, Beta_0_star, W_0_star_V, nu_0_star, seed=None)

            # user hyperparameters
            # U_average=np.transpose(np.array(np.sum(U_old, axis=1)/N, ndmin=2)) #the np.array and np.transpose are needed for it to be a column vector
            U_average = np.sum(U_old, axis=1) / N  # in this way it is a 1d array!!  #D-long
            S_bar_U = np.dot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD
            # S_bar_U = npdot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD

            mu_0_star_U = (Beta_0 * mu_0 + N * U_average) / (Beta_0 + N)
            W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
            W_0_star_U = np.linalg.inv(W_0_star_U_inv)
            # W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * npdot(
            #     np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
            # W_0_star_U = nplinalginv(W_0_star_U_inv)
            mu_U, Lambda_U, cov_U = Normal_Wishart(mu_0_star_U, Beta_0_star, W_0_star_U, nu_0_star, seed=None)

            # print (S_bar_U.shape, S_bar_V.shape)
            # print (np.dot(np.transpose(np.array(mu_0-U_average, ndmin=2)),np.array((mu_0-U_average), ndmin=2).shape))

            """SAMPLE THEN USER FEATURES (possibly in parallel):"""

            # U_new = np.array([])  # define the new stuff.
            # V_new = np.array([])

            # num_cores = multiprocessing.cpu_count() - 2
            # resultsU, mu_U, lam_U_inv = zip(*Parallel(n_jobs=num_cores)(delayed(SampleUser)(i, V_old, mu_U, Lambda_U) for i in range(N)))
            # U_new = np.transpose(np.reshape(resultsU, (N, D)))
            # resultsV, mu_V, lam_V_inv = zip(*Parallel(n_jobs=num_cores)(delayed(SampleFeature)(U_new,j, mu_V, Lambda_V) for j in range(M)))
            # V_new = np.transpose(np.reshape(resultsV, (M, D)))

            num_cores = multiprocessing.cpu_count() - 2
            resultsU = Parallel(n_jobs=num_cores)(delayed(SampleUser)(i, V_old, mu_U, Lambda_U) for i in range(N))
            U_new = np.transpose(np.reshape(resultsU, (N, D)))
            resultsV = Parallel(n_jobs=num_cores)(delayed(SampleFeature)(U_new,j, mu_V, Lambda_V) for j in range(M))
            V_new = np.transpose(np.reshape(resultsV, (M, D)))

            # resultsU = []
            # for iN in range(N):
            #     tresultsU, tmu_U, tlam_U_inv = SampleUser(iN, V_old, mu_U, Lambda_U)
            #     resultsU.append(tresultsU)
            # U_new = np.transpose(np.reshape(resultsU, (N, D)))
            #
            # resultsV = []
            # for iM in range(M):
            #     tresultsV, tmu_V, tlam_V_inv = SampleFeature(U_new,iM, mu_V, Lambda_V)
            #     resultsV.append(tresultsV)
            # V_new = np.transpose(np.reshape(resultsV, (M, D)))

            # for i in range(N):  # loop over the users
            #     # first compute the parameters of the distribution
            #     Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            #     mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            #     for j in range(M):  # loop over the movies
            #         if ranked(i, j):  # only if movie j has been ranked by user i!
            #             Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
            #                                              np.array((V_old[:, j]), ndmin=2))  # CHECK
            #             mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
            #             # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!
            #
            #     Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
            #     Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)
            #
            #     ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
            #     mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U, mu_U)
            #     mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
            #     # extract now the U values!
            #     U_new = np.append(U_new, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))
            #
            # # you need to reshape U_new and transpose it!!
            # U_new = np.transpose(np.reshape(U_new, (N, D)))
            # # print("U new",U_new)
            # # exit()
            # """SAMPLE THEN MOVIE FEATURES (possibly in parallel):"""
            #
            # for j in range(M):
            #     Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            #     mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            #     for i in range(N):  # loop over the movies
            #         if ranked(i, j):
            #             Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(U_new[:, i], ndmin=2)),
            #                                              np.array((U_new[:, i]), ndmin=2))
            #             mu_i_star_1 = U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
            #             # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!
            #
            #     Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
            #     Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)
            #
            #     mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
            #     mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            #     V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))
            #
            # # you need to reshape U_new and transpose it!!
            # V_new = np.transpose(np.reshape(V_new, (M, D)))

            # save U_new and V_new in U_old and V_old for next iteration:
            U_old = np.array(U_new)
            V_old = np.array(V_new)

            # print (V_new.shape)
            # print (V_new.shape, U_new.shape)

            if t > initial_cutoff:  # initial_cutoff is needed to discard the initial transient
                R_step = np.dot(np.transpose(U_new), V_new) # RECONSTRUCT R from U and V
                R_step = np.clip(R_step, 0, 1)
                # for i in range(N):  # reduce all the predictions to the correct ratings range.
                #     for j in range(M):
                #         if R_step[i, j] > highest_rating:
                #             R_step[i, j] = highest_rating
                #         elif R_step[i, j] < lowest_rating:
                #             R_step[i, j] = lowest_rating

                R_predict = (R_predict * (t - initial_cutoff - 1) + R_step) / (t - initial_cutoff)
                # train_err = 0  # initialize the errors.
                # test_err = 0

                # print("inner R_predict:", R_predict)
                # print("inner R_step:", R_step)

                # compute now the RMSE on the train dataset:
                # for i in range(N):
                #     for j in range(M):
                #         if ranked(i, j):
                #             train_err = train_err + (R_predict[i, j] - R[i, j]) ** 2
                # train_err_list.append(np.sqrt(train_err / pairs_train))
                #
                # print("Training RMSE at iteration ", t - initial_cutoff, " :   ", "{:.4}".format(train_err_list[-1]))
                # # compute now the RMSE on the test dataset:
                # for i in range(N):
                #     for j in range(M):
                #         if ranked_test(i, j):
                #             test_err = test_err + (R_predict[i, j] - R_test[i, j]) ** 2
                # test_err_list.append(np.sqrt(test_err / pairs_test))
                # print("Test RMSE at iteration ", t - initial_cutoff, " :   ", "{:.4}".format(test_err_list[-1]))
                #
                # train_epoch_list.append(t)
                #
                # # row = pd.DataFrame.from_items([('step', t), ('train_err', train_err), ('test_err', test_err)])
                # # results = results.append(row)  # save results at every iteration:
                # results = pd.DataFrame.from_items([('step', train_epoch_list), ('train_err', train_err_list), ('test_err', test_err_list)])
                # if save_file:
                #     results.to_csv(output_file)
        # for test in range(40):
        #     new_U = [multivariate_normal(mu_U[ii], lam_U_inv[ii]) for ii in range(N)]
        #     new_V = [multivariate_normal(mu_V[ii], lam_V_inv[ii]) for ii in range(M)]
        #     new_U = np.transpose(np.reshape(new_U, (N, D)))
        #     new_V = np.transpose(np.reshape(new_V, (M, D)))
        #     tmpR = np.dot(np.transpose(new_U), new_V)  # RECONSTRUCT R from U and V
        #     tmpR = np.clip(tmpR, 0, 1)
        #     if tmpR[-1,-1] > R_predict[-1,-1]:
        #         R_predict = tmpR
        # return R_predict, train_err_list, test_err_list, train_epoch_list
        U_in = U_new
        V_in = V_new
        return R_predict, train_err_list

    def BPMF(self, R, R_test, U_in, V_in, T, D, initial_cutoff, lowest_rating, highest_rating, in_alpha, output_file,
             mu_0=None, Beta_0=None, W_0=None, nu_0=None, missing_mask = -1, save_file=True):
        """
        R is the ranking matrix (NxM, N=#users, M=#movies); we are assuming that R[i,j]=0 means that user i has not ranked movie j
        R_test is the ranking matrix that contains test values. Same assumption as above.
        U_in, V_in are the initial values for the MCMC procedure.
        T is the number of steps.
        D is the number of hidden features that are assumed in the model.

        mu_0 is the average vector used in sampling the multivariate normal variable
        Beta_0 is a coefficient (?)
        W_0 is the DxD scale matrix in the Wishart sampling
        nu_0 is the number of degrees of freedom used in the Wishart sampling.

        U matrices are DxN, while V matrices are DxM.

        If save_file=True, this function internally saves the file at each iteration; this results in a different file for each value
        of D and is useful when the algorithm may stop during the execution.
        """

        # @njit
        def ranked(i, j):  # function telling if user i ranked movie j in the train dataset.
            if R[i, j] != missing_mask:
                return True
            else:
                return False

        N = R.shape[0]
        M = R.shape[1]

        R_predict = np.zeros((N, M))
        U_old = np.array(U_in)
        V_old = np.array(V_in)

        train_err_list = []


        # initialize now the hierarchical priors:
        alpha = in_alpha  # observation noise, they put it = 2 in the paper

        # SET THE DEFAULT VALUES for Wishart distribution
        # we assume that parameters for both U and V are the same.

        if mu_0 is None:
            mu_0 = np.zeros(D)
        if nu_0 is None:
            nu_0 = D
        if Beta_0 is None:
            Beta_0 = 2
        if W_0 is None:
            W_0 = np.eye(D)

        # results = pd.DataFrame(columns=['step', 'train_err', 'test_err'])

        # parameters common to both distributions:
        Beta_0_star = Beta_0 + N
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)  # compute the inverse once and for all

        def SampleUser(i, inV, inMu_U, inLambda_U):
            Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for j in range(M):  # loop over the movies
                if ranked(i, j):  # only if movie j has been ranked by user i!
                    Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(inV[:, j], ndmin=2)),
                                                     np.array((inV[:, j]), ndmin=2))  # CHECK
                    # Lambda_U_2 = Lambda_U_2 + npdot(nptranspose(np.array(inV[:, j], ndmin=2)),
                    #                                  np.array((inV[:, j]), ndmin=2))  # CHECK
                    mu_i_star_1 = inV[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_i_star_U = inLambda_U + alpha * Lambda_U_2
            Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

            ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
            mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_U, inMu_U)
            mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
            output = multivariate_normal(mu_i_star, Lambda_i_star_U_inv)

            return output

        def SampleFeature(input_U_new, j, inMu_V, inLambda_V):
            Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for i in range(N):  # loop over the movies
                if ranked(i, j):
                    Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(input_U_new[:, i], ndmin=2)),
                                                     np.array((input_U_new[:, i]), ndmin=2))
                    # Lambda_V_2 = Lambda_V_2 + npdot(np.transpose(np.array(input_U_new[:, i], ndmin=2)),
                    #                                  np.array((input_U_new[:, i]), ndmin=2))
                    mu_i_star_1 = input_U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_j_star_V = inLambda_V + alpha * Lambda_V_2
            Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)
            # Lambda_j_star_V_inv = nplinalginv(Lambda_j_star_V)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_V, inMu_V)
            mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            # mu_i_star_part = alpha * mu_i_star_1 + np.dot(inLambda_V, inMu_V)
            # mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            output = multivariate_normal(mu_j_star, Lambda_j_star_V_inv)

            # return output, mu_j_star, Lambda_j_star_V_inv
            return output

        for t in range(T):
            # print("Step ", t)
            # FIRST SAMPLE THE HYPERPARAMETERS, conditioned on the present step user and movie feature matrices U_t and V_t:

            # movie hyperparameters:
            V_average = np.sum(V_old, axis=1) / N  # in this way it is a 1d array!!
            S_bar_V = np.dot(V_old, np.transpose(V_old)) / N
            # S_bar_V = npdot(V_old, np.transpose(V_old)) / N

            mu_0_star_V = (Beta_0 * mu_0 + N * V_average) / (Beta_0 + N)
            W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
            # W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * npdot(
            #     np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
            W_0_star_V = np.linalg.inv(W_0_star_V_inv)
            # W_0_star_V = nplinalginv(W_0_star_V_inv)
            mu_V, Lambda_V, cov_V = Normal_Wishart(mu_0_star_V, Beta_0_star, W_0_star_V, nu_0_star, seed=None)

            # user hyperparameters
            # U_average=np.transpose(np.array(np.sum(U_old, axis=1)/N, ndmin=2)) #the np.array and np.transpose are needed for it to be a column vector
            U_average = np.sum(U_old, axis=1) / N  # in this way it is a 1d array!!  #D-long
            S_bar_U = np.dot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD
            # S_bar_U = npdot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD

            mu_0_star_U = (Beta_0 * mu_0 + N * U_average) / (Beta_0 + N)
            W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
            W_0_star_U = np.linalg.inv(W_0_star_U_inv)
            # W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * npdot(
            #     np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
            # W_0_star_U = nplinalginv(W_0_star_U_inv)
            mu_U, Lambda_U, cov_U = Normal_Wishart(mu_0_star_U, Beta_0_star, W_0_star_U, nu_0_star, seed=None)

            # print (S_bar_U.shape, S_bar_V.shape)
            # print (np.dot(np.transpose(np.array(mu_0-U_average, ndmin=2)),np.array((mu_0-U_average), ndmin=2).shape))

            """SAMPLE THEN USER FEATURES (possibly in parallel):"""

            U_new = np.array([])  # define the new stuff.
            V_new = np.array([])

            # num_cores = multiprocessing.cpu_count() - 2
            # resultsU = Parallel(n_jobs=num_cores)(delayed(SampleUser)(i, V_old, mu_U, Lambda_U) for i in range(N))
            # U_new = np.transpose(np.reshape(resultsU, (N, D)))
            # resultsV = Parallel(n_jobs=num_cores)(delayed(SampleFeature)(U_new,j, mu_V, Lambda_V) for j in range(M))
            # V_new = np.transpose(np.reshape(resultsV, (M, D)))

            # resultsU = []
            # for iN in range(N):
            #     tresultsU, tmu_U, tlam_U_inv = SampleUser(iN, V_old, mu_U, Lambda_U)
            #     resultsU.append(tresultsU)
            # U_new = np.transpose(np.reshape(resultsU, (N, D)))
            #
            # resultsV = []
            # for iM in range(M):
            #     tresultsV, tmu_V, tlam_V_inv = SampleFeature(U_new,iM, mu_V, Lambda_V)
            #     resultsV.append(tresultsV)
            # V_new = np.transpose(np.reshape(resultsV, (M, D)))

            for i in range(N):  # loop over the users
                # first compute the parameters of the distribution
                Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                for j in range(M):  # loop over the movies
                    if ranked(i, j):  # only if movie j has been ranked by user i!
                        Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
                                                         np.array((V_old[:, j]), ndmin=2))  # CHECK
                        mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                        # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
                Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

                ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
                mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U, mu_U)
                mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
                # extract now the U values!
                U_new = np.append(U_new, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))

            # you need to reshape U_new and transpose it!!
            U_new = np.transpose(np.reshape(U_new, (N, D)))
            # print("U new",U_new)
            # exit()
            """SAMPLE THEN MOVIE FEATURES (possibly in parallel):"""

            for j in range(M):
                Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                for i in range(N):  # loop over the movies
                    if ranked(i, j):
                        Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(U_new[:, i], ndmin=2)),
                                                         np.array((U_new[:, i]), ndmin=2))
                        mu_i_star_1 = U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                        # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
                Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)

                mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
                mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
                V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))

            # you need to reshape U_new and transpose it!!
            V_new = np.transpose(np.reshape(V_new, (M, D)))

            # save U_new and V_new in U_old and V_old for next iteration:
            U_old = np.array(U_new)
            V_old = np.array(V_new)

            # print (V_new.shape)
            # print (V_new.shape, U_new.shape)

            if t > initial_cutoff:  # initial_cutoff is needed to discard the initial transient
                R_step = np.dot(np.transpose(U_new), V_new) # RECONSTRUCT R from U and V
                R_step = np.clip(R_step, 0, 1)
                # for i in range(N):  # reduce all the predictions to the correct ratings range.
                #     for j in range(M):
                #         if R_step[i, j] > highest_rating:
                #             R_step[i, j] = highest_rating
                #         elif R_step[i, j] < lowest_rating:
                #             R_step[i, j] = lowest_rating

                R_predict = (R_predict * (t - initial_cutoff - 1) + R_step) / (t - initial_cutoff)
                # train_err = 0  # initialize the errors.
                # test_err = 0

                # print("inner R_predict:", R_predict)
                # print("inner R_step:", R_step)

                # compute now the RMSE on the train dataset:
                # for i in range(N):
                #     for j in range(M):
                #         if ranked(i, j):
                #             train_err = train_err + (R_predict[i, j] - R[i, j]) ** 2
                # train_err_list.append(np.sqrt(train_err / pairs_train))
                #
                # print("Training RMSE at iteration ", t - initial_cutoff, " :   ", "{:.4}".format(train_err_list[-1]))

        # for test in range(40):
        #     new_U = [multivariate_normal(mu_U[ii], lam_U_inv[ii]) for ii in range(N)]
        #     new_V = [multivariate_normal(mu_V[ii], lam_V_inv[ii]) for ii in range(M)]
        #     new_U = np.transpose(np.reshape(new_U, (N, D)))
        #     new_V = np.transpose(np.reshape(new_V, (M, D)))
        #     tmpR = np.dot(np.transpose(new_U), new_V)  # RECONSTRUCT R from U and V
        #     tmpR = np.clip(tmpR, 0, 1)
        #     if tmpR[-1,-1] > R_predict[-1,-1]:
        #         R_predict = tmpR
        # return R_predict, train_err_list, test_err_list, train_epoch_list
        U_in = U_new
        V_in = V_new
        return R_predict, train_err_list

    def ProposedBPMF(self, R, R_test, U_in, V_in, T, D, initial_cutoff, lowest_rating, highest_rating, in_alpha, numSamples, output_file,
             mu_0=None, Beta_0=None, W_0=None, nu_0=None, missing_mask = -1, save_file=True):
        """
        R is the ranking matrix (NxM, N=#users, M=#movies); we are assuming that R[i,j]=0 means that user i has not ranked movie j
        R_test is the ranking matrix that contains test values. Same assumption as above.
        U_in, V_in are the initial values for the MCMC procedure.
        T is the number of steps.
        D is the number of hidden features that are assumed in the model.

        mu_0 is the average vector used in sampling the multivariate normal variable
        Beta_0 is a coefficient (?)
        W_0 is the DxD scale matrix in the Wishart sampling
        nu_0 is the number of degrees of freedom used in the Wishart sampling.

        U matrices are DxN, while V matrices are DxM.

        If save_file=True, this function internally saves the file at each iteration; this results in a different file for each value
        of D and is useful when the algorithm may stop during the execution.
        """

        # @njit
        def ranked(i, j):  # function telling if user i ranked movie j in the train dataset.
            if R[i, j] != missing_mask:
                return True
            else:
                return False

        N = R.shape[0]
        M = R.shape[1]

        R_predict = np.zeros((N, M))
        U_old = np.array(U_in)
        V_old = np.array(V_in)

        train_err_list = []


        # initialize now the hierarchical priors:
        alpha = in_alpha  # observation noise, they put it = 2 in the paper

        # SET THE DEFAULT VALUES for Wishart distribution
        # we assume that parameters for both U and V are the same.

        if mu_0 is None:
            mu_0 = np.zeros(D)
        if nu_0 is None:
            nu_0 = D
        if Beta_0 is None:
            Beta_0 = 2
        if W_0 is None:
            W_0 = np.eye(D)

        # results = pd.DataFrame(columns=['step', 'train_err', 'test_err'])

        # parameters common to both distributions:
        Beta_0_star = Beta_0 + N
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)  # compute the inverse once and for all

        # Output N new sampled Rs
        Rs = []

        for t in range(T):
            # print("Step ", t)
            # FIRST SAMPLE THE HYPERPARAMETERS, conditioned on the present step user and movie feature matrices U_t and V_t:

            # movie hyperparameters:
            V_average = np.sum(V_old, axis=1) / N  # in this way it is a 1d array!!
            S_bar_V = np.dot(V_old, np.transpose(V_old)) / N

            mu_0_star_V = (Beta_0 * mu_0 + N * V_average) / (Beta_0 + N)
            W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
            W_0_star_V = np.linalg.inv(W_0_star_V_inv)
            mu_V, Lambda_V, cov_V = Normal_Wishart(mu_0_star_V, Beta_0_star, W_0_star_V, nu_0_star, seed=None)

            # user hyperparameters
            # U_average=np.transpose(np.array(np.sum(U_old, axis=1)/N, ndmin=2)) #the np.array and np.transpose are needed for it to be a column vector
            U_average = np.sum(U_old, axis=1) / N  # in this way it is a 1d array!!  #D-long
            S_bar_U = np.dot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD
            # S_bar_U = npdot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD

            mu_0_star_U = (Beta_0 * mu_0 + N * U_average) / (Beta_0 + N)
            W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * np.dot(np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
            W_0_star_U = np.linalg.inv(W_0_star_U_inv)
            mu_U, Lambda_U, cov_U = Normal_Wishart(mu_0_star_U, Beta_0_star, W_0_star_U, nu_0_star, seed=None)

            """SAMPLE THEN USER FEATURES (possibly in parallel):"""

            U_new = np.array([])  # define the new stuff.
            V_new = np.array([])

            for i in range(N):  # loop over the users
                # first compute the parameters of the distribution
                Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                for j in range(M):  # loop over the movies
                    if ranked(i, j):  # only if movie j has been ranked by user i!
                        Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
                                                         np.array((V_old[:, j]), ndmin=2))  # CHECK
                        mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                        # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
                Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

                ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
                mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U, mu_U)
                mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
                # extract now the U values!
                U_new = np.append(U_new, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))

            # you need to reshape U_new and transpose it!!
            U_new = np.transpose(np.reshape(U_new, (N, D)))
            """SAMPLE THEN MOVIE FEATURES (possibly in parallel):"""

            for j in range(M):
                Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                for i in range(N):  # loop over the movies
                    if ranked(i, j):
                        Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(U_new[:, i], ndmin=2)),
                                                         np.array((U_new[:, i]), ndmin=2))
                        mu_i_star_1 = U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                        # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
                Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)

                mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
                mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
                V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))

            # you need to reshape U_new and transpose it!!
            V_new = np.transpose(np.reshape(V_new, (M, D)))

            # save U_new and V_new in U_old and V_old for next iteration:
            U_old = np.array(U_new)
            V_old = np.array(V_new)

            # Sample N new matrix R
            if t == T-1:
                # print("Sampling!!!!")
                for ii in range(numSamples):
                    Utmp = np.array([])  # define the new stuff.
                    Vtmp = np.array([])

                    for i in range(N):  # loop over the users
                        # first compute the parameters of the distribution
                        Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                        mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                        for j in range(M):  # loop over the movies
                            if ranked(i, j):  # only if movie j has been ranked by user i!
                                Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
                                                                 np.array((V_old[:, j]), ndmin=2))  # CHECK
                                mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                                # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                        Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
                        Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

                        ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
                        mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U, mu_U)
                        mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
                        # extract now the U values!
                        Utmp = np.append(Utmp, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))

                    # you need to reshape U_new and transpose it!!
                    Utmp = np.transpose(np.reshape(Utmp, (N, D)))

                    for j in range(M):
                        Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
                        mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
                        for i in range(N):  # loop over the movies
                            if ranked(i, j):
                                Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(Utmp[:, i], ndmin=2)),
                                                                 np.array((Utmp[:, i]), ndmin=2))
                                mu_i_star_1 = Utmp[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                                # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

                        Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
                        Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)

                        mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
                        mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
                        Vtmp = np.append(Vtmp, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))

                    # you need to reshape U_new and transpose it!!
                    Vtmp = np.transpose(np.reshape(Vtmp, (M, D)))

                    Rtmp = np.dot(np.transpose(Utmp), Vtmp)  # RECONSTRUCT R from U and V
                    Rtmp = np.clip(Rtmp, 0, 1)
                    Rpred = (Rtmp * (t - initial_cutoff - 1) + Rtmp) / (t - initial_cutoff)
                    Rs.append(Rpred.tolist())

            if t > initial_cutoff:  # initial_cutoff is needed to discard the initial transient
                R_step = np.dot(np.transpose(U_new), V_new) # RECONSTRUCT R from U and V
                R_step = np.clip(R_step, 0, 1)

                R_predict = (R_predict * (t - initial_cutoff - 1) + R_step) / (t - initial_cutoff)

        U_in = U_new
        V_in = V_new
        return R_predict, train_err_list, Rs