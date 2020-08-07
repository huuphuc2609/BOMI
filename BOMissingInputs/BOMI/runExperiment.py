import torch
import gpytorch
import time
import math
import datetime
import pickle
import sys
parent_folder = 'C:\Deakin\pluong\BOMissingInputs' # Path to the project's root
sys.path.append(parent_folder)

import pandas as pd
from BOMI.BOGPyTorch.GPTmodels import ExactGPModel
from scipy.stats import mode

from BOMI.MyBPMF.MatrixFactorization import myBPMF
from BOMI.ndfunction import *

from sklearn.neighbors import KernelDensity

from missingpy import KNNImputer

mySeed = 9
np.random.seed(mySeed)
torch.random.manual_seed(mySeed)

# Define BO class
class BOTorch():
    def __init__(self):
        return

    @staticmethod
    def optimizeHyperparamaters(in_model, train_x, train_y, optimizerAlgo, trainingIter):
        # Set train data
        train_x.cuda()
        train_y.cuda()
        in_model.cuda()
        in_model.set_train_data(train_x, train_y)
        # Find optimal model hyperparameters
        in_model.train()
        in_model.likelihood.train()
        # Use the adam optimizer
        if optimizerAlgo == "Adam":
            optimizer = torch.optim.Adam([{'params': in_model.parameters()}, ],
                                         lr=0.1)  # Includes GaussianLikelihood parameters
        elif optimizerAlgo == "SGD":
            optimizer = torch.optim.SGD([{'params': in_model.parameters()}, ],
                                        lr=0.1)  # Includes GaussianLikelihood parameters
        elif optimizerAlgo == "LBFGS":
            optimizer = torch.optim.LBFGS([{'params': in_model.parameters()}, ], lr=0.1, history_size=50, max_iter=10,
                                          line_search_fn=True)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(in_model.likelihood, in_model)
        training_iter = trainingIter

        # LBFGS
        # define closure
        if optimizerAlgo == "LBFGS":
            def closure():
                optimizer.zero_grad()
                output = in_model(train_x)
                loss = -mll(output, train_y)
                return loss

            loss = closure()
            loss.backward()

            for i in range(training_iter):
                # perform step and update curvature
                loss = optimizer.step(closure)
                loss.backward()

        elif optimizerAlgo == "SGD" or optimizerAlgo == "Adam":
            # SGD & Adam
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = in_model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()


    @staticmethod
    def posterior_predict(in_x, in_model):
        in_model.eval()
        in_model.likelihood.eval()
        # Calculate posterior predictions
        f_preds = in_model(in_x)
        # Take mean and variance
        tmpMean = f_preds.mean
        tmpVar = f_preds.variance
        # Get from tensors
        mean = tmpMean.detach().numpy()
        var = tmpVar.detach().numpy()
        return mean, var

    @staticmethod
    def posterior_predict_scalar(in_x, in_model):
        in_model.eval()
        in_model.likelihood.eval()
        mean = []
        var = []
        for point in in_x:
            # Push to tensor
            point = torch.tensor([point.item()]).cuda()
            # Calculate posterior predictions
            f_preds = in_model(point)
            # Take mean and variance
            tmpMean = f_preds.mean
            tmpVar = f_preds.variance
            # Get from tensors
            mean.append(tmpMean.item())
            var.append(tmpVar.item())
        return mean, var

    @staticmethod
    def gpucb(in_x, in_model, in_Beta):
        in_model.eval()
        in_model.likelihood.eval()
        x = torch.tensor(in_x, dtype=torch.float32).cuda()
        # Calculate posterior predictions
        f_preds = in_model(x)
        # Take mean and variance
        tmpMean = f_preds.mean
        tmpVar = f_preds.variance
        # Calculate acquisition
        tmpAcq = tmpMean + torch.tensor(in_Beta).cuda() * tmpVar
        # Get from tensors
        acq = tmpAcq.clone().detach().cpu().numpy()
        tmpMean = tmpMean.clone().detach().cpu().numpy()
        tmpVar = tmpVar.clone().detach().cpu().numpy()
        return acq, tmpMean, tmpVar

    @staticmethod
    def gpucb_scalar(in_x, in_model, in_Beta):
        in_model.eval()
        in_model.likelihood.eval()
        for i in range(len(in_x)):
            in_x[i] = float(in_x[i])
        x = torch.tensor([in_x], dtype=torch.float32)
        # print("type x:", x)
        # Calculate posterior predictions
        f_preds = in_model(x)
        # Take mean and variance
        tmpMean = f_preds.mean
        tmpVar = f_preds.variance
        # Calculate acquisition
        tmpAcq = tmpMean + torch.tensor(in_Beta) * tmpVar
        # Get from tensors
        acq = tmpAcq.detach().cpu().numpy()
        return acq

# Input arguments
input_method = sys.argv[1] # method = "Drop", "Suggest", "BPMF", "BOMI", "Mean", "Mode", "KNN", "uGP"
input_function = sys.argv[2] # Black-box objective functions, see file 'ndfunction.py'
input_numGPs = int(sys.argv[3]) # Default: 5 (GPs)
input_in_alpha = float(sys.argv[4]) # Default: 1e2 - BPMF parameter
input_missingRate = float(sys.argv[5]) # 0.25
input_missingNoise = float(sys.argv[6]) # 0.05

# Select true objective function
missRate = input_missingRate
missNoise = input_missingNoise
if input_function == "Eggholder2d":
    myfunc = Eggholder(2, missRate, missNoise, False)
elif input_function == "Schubert4d":
    myfunc = Schubert_Nd(4, missRate,missNoise, False)
elif input_function == "Alpine5d":
    myfunc = Alpine_Nd(5,missRate,missNoise,False)
elif input_function == "Schwefel5d":
    myfunc = Schwefel_Nd(5, missRate,missNoise, False)

# GP-UCB parameters
scaleBeta = 0.2
delt = 0.1
a = 1.0
b = 1.0
dim = myfunc.input_dim
r = 1.0

BetaMu = 1.0
BetaSigma = 1.0

# Experiments settings
Runs = 10
if input_function == "Eggholder2d":
    NumIterations = 61
elif input_function == "Schubert4d":
    NumIterations = 121
elif input_function == "Alpine5d" or input_function == "Schwefel5d":
    NumIterations = 161

log_iter = ""
log_iter_point = ""
log_run = ""

# Select method
method = input_method
isMulMethod = True
numGPs = input_numGPs
GPHypers_optim = "Adam"
GPoptimIter = 50
totalTimeStart = time.time()
for run in range(0, Runs):
    # Read data from file
    if input_function == "Eggholder2d":
        dfX = pd.read_csv(r'data/Eggholder2d/Eggholder2dX_' + str(run) + '.csv')
    elif input_function == "Schubert4d":
        dfX = pd.read_csv(r'data/Schubert4d/SchubertNd4dX_' + str(run) + '.csv')
    elif input_function == "Alpine5d":
        dfX = pd.read_csv(r'data/Alpine5d/AlpineNd5dX_' + str(run) + '.csv')
    elif input_function == "Schwefel5d":
        dfX = pd.read_csv(r'data/Schwefel5d/SchwefelNd5dX_' + str(run) + '.csv')

    inX = dfX.values
    Xori = inX.tolist().copy()

    if input_function == "Eggholder2d":
        dfY = pd.read_csv(r'data/Eggholder2d/EggholderY.csv')
    elif input_function == "Schubert4d":
        dfY = pd.read_csv(r'data/Schubert4d/SchubertNdY.csv')
    elif input_function == "Alpine5d":
        dfY = pd.read_csv(r'data/Alpine5d/AlpineNdY.csv')
    elif input_function == "Schwefel5d":
        dfY = pd.read_csv(r'data/Schwefel5d/SchwefelNdY.csv')

    inY = dfY.values
    inY = inY.squeeze()
    Yori = inY.tolist().copy()

    na = 0.0
    nb = 1.0
    minY = np.min(Yori)
    maxY = np.max(Yori)
    normalizedY = [[(((nb - na) * (tmpY - minY) / (maxY - minY) + na))] for tmpY in Yori]

    normalizedX = [myfunc.normalize(xi) for xi in Xori]

    R = np.append(normalizedX, normalizedY, axis=1)
    R[np.isnan(R)] = -1

    # Arrays of observations after dropping/removing
    dropNX = []
    dropNY = []
    dropYori = []
    dropXori = []
    for i in range(len(normalizedX)):
        if -1 not in R[i]:
            dropNX.append(normalizedX[i])
            dropNY.append(normalizedY[i])
            dropXori.append(Xori[i])
            dropYori.append(Yori[i])

    if method == "Drop" or method == "Suggest" or method == "uGP":
        in_X = dropXori
        in_Y = dropYori
        minY = np.min(in_Y)
        maxY = np.max(in_Y)
        if minY == maxY:
            minY -= 0.01
        n_in_Y = [(((nb - na) * (tmpY - minY) / (maxY - minY) + na)) for tmpY in in_Y]
        n_in_X = [myfunc.normalize(xi) for xi in in_X]
    else:
        in_X = Xori
        in_Y = Yori
        n_in_X = normalizedX
        n_in_Y = normalizedY

    # Initialize likelihood and model
    train_x = torch.tensor(n_in_X, dtype=torch.float32).cuda()
    train_y = torch.tensor(n_in_Y, dtype=torch.float32).cuda()
    myGP = ExactGPModel(train_x, train_y).cuda()

    # Init BO objects:
    myBO = BOTorch()

    if method == "BPMF" or method == "BOMI":
        R_in = np.append(n_in_X, n_in_Y, axis=1)
        R_in[np.isnan(R_in)] = -1
        D = 15
        (N, M) = R.shape
        T = 40

        beta0 = None
        myBPMFObject = myBPMF()

    in_Guessed_X = dropXori
    in_Guessed_Y = dropYori
    in_Guessed_nY = [(((nb - na) * (tmpY - np.min(in_Guessed_Y)) / (np.max(in_Guessed_Y) - np.min(in_Guessed_Y)) + na)) for tmpY in in_Guessed_Y]
    in_Guessed_nX = [myfunc.normalize(xi) for xi in in_Guessed_X]

    ####################### BO loops #######################
    myfunc.numCalls = 0
    BOstart_time = time.time()
    for ite in range(NumIterations):
        precalBetaT = 2.0 * np.log((ite + 1) * (ite + 1) * math.pi ** 2 / (3 * delt)) + 2 * dim * np.log(
            (ite + 1) * (ite + 1) * dim * b * r * np.sqrt(np.log(4 * dim * a / delt)))
        BetaT = np.sqrt(precalBetaT) * scaleBeta

        # Train the GP model
        # BPMF require filling missing values first before feeding into the GP
        if method == "BPMF":
            R_in = np.append(n_in_X, n_in_Y, axis=1)
            R_in[np.isnan(R_in)] = -1
            newX = []
            newY = []

            (N, M) = R_in.shape
            U_in = np.zeros((D, N))
            V_in = np.zeros((D, M))

            in_alpha = input_in_alpha * r
            R_pred, train_err_list, Rs = myBPMFObject.ProposedBPMF(R_in, R_in, U_in, V_in, T, D,
                                                                   initial_cutoff=0, lowest_rating=0.0,
                                                                   highest_rating=1.0, in_alpha=in_alpha,
                                                                   numSamples=numGPs,
                                                                   Beta_0=beta0, output_file=None,
                                                                   missing_mask=-1,
                                                                   save_file=False)
            # Use the new predicted matrix as input training data for the GP
            res = R_pred.copy()
            newX = np.delete(res, myfunc.input_dim, axis=1)
            newY = (np.delete(res, 0, axis=1)).tolist()
            for _ in range(myfunc.input_dim - 2):
                newY = (np.delete(newY, 0, axis=1)).tolist()
            newY = ((np.delete(newY, 0, axis=1)).reshape(1, -1)).tolist()

            train_x = torch.tensor(newX, dtype=torch.float32).cuda()
            train_y = torch.tensor(newY, dtype=torch.float32).cuda()

            myGP = ExactGPModel(train_x, train_y).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)
        elif method == "BOMI":
            BPMFstart_time = time.time()

            R_in = np.append(n_in_X, n_in_Y, axis=1)
            R_in[np.isnan(R_in)] = -1

            newX = []
            newY = []

            (N, M) = R_in.shape
            U_in = np.zeros((D, N))
            V_in = np.zeros((D, M))

            in_alpha = input_in_alpha*r
            R_pred, train_err_list, Rs = myBPMFObject.ProposedBPMF(R_in, R_in, U_in, V_in, T, D,
                                                                   initial_cutoff=0, lowest_rating=0.0,
                                                                   highest_rating=1.0, in_alpha=in_alpha,
                                                                   numSamples=numGPs,
                                                                   Beta_0=beta0, output_file=None,
                                                                   missing_mask=-1,
                                                                   save_file=False)

            BPMFstop_time = time.time()

            PRstart_time = time.time()
            idxs = np.where(R_in == -1)
            for ii in range(numGPs):
                tmpR = R_in.copy()
                for iii in range(len(idxs[0])):
                    tmpR[idxs[0][iii], idxs[1][iii]] = Rs[ii][idxs[0][iii]][idxs[1][iii]]
                Rs[ii] = tmpR

            PRstop_time = time.time()

            # Use the new predicted matrix as input training data for the GP
            BuildGPstart_time = time.time()
            GPs = []
            for ii in range(numGPs):
                res = Rs[ii].copy()
                newX = np.delete(res, myfunc.input_dim, axis=1)
                newY = (np.delete(res, 0, axis=1)).tolist()
                for _ in range(myfunc.input_dim - 2):
                    newY = (np.delete(newY, 0, axis=1)).tolist()
                newY = ((np.delete(newY, 0, axis=1)).reshape(1, -1)).tolist()

                train_x = torch.tensor(newX, dtype=torch.float32).cuda()
                train_y = torch.tensor(newY, dtype=torch.float32).cuda()
                tmpGP = ExactGPModel(train_x, train_y).cuda()

                myBO.optimizeHyperparamaters(tmpGP, train_x, train_y, GPHypers_optim, GPoptimIter)
                GPs.append(tmpGP)

            BuildGPstop_time = time.time()
        elif method == "Mean":

            R_in = np.append(n_in_X, n_in_Y, axis=1)
            R_in[np.isnan(R_in)] = -1
            idxs = np.where(R_in == -1)

            tmpR = R_in.copy()
            meansR = np.mean(tmpR, axis=0)
            for iii in range(len(idxs[0])):
                tmpR[idxs[0][iii], idxs[1][iii]] = meansR[idxs[1][iii]]

            # Use the new predicted matrix as input training data for the GP
            BuildGPstart_time = time.time()
            GPs = []

            res = tmpR

            newX = np.delete(res, myfunc.input_dim, axis=1)
            newY = (np.delete(res, 0, axis=1)).tolist()

            n_in_X = (newX.tolist()).copy()
            n_in_Y = newY.copy()

            for _ in range(myfunc.input_dim - 2):
                newY = (np.delete(newY, 0, axis=1)).tolist()
            newY = ((np.delete(newY, 0, axis=1)).reshape(1, -1)).tolist()

            train_x = torch.tensor(newX, dtype=torch.float32).cuda()
            train_y = torch.tensor(newY, dtype=torch.float32).cuda()
            myGP = ExactGPModel(train_x, train_y).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)

            BuildGPstop_time = time.time()
            print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
        elif method == "Mode":

            R_in = np.append(n_in_X, n_in_Y, axis=1)
            RforIdx = R_in.copy()
            RforIdx[np.isnan(RforIdx)] = -1

            idxs = np.where(RforIdx == -1)

            tmpR = R_in.copy()
            modesR = mode(tmpR, axis=0, nan_policy='omit')[0][0]
            for iii in range(len(idxs[0])):
                tmpR[idxs[0][iii], idxs[1][iii]] = modesR[idxs[1][iii]]

            # Use the new predicted matrix as input training data for the GP
            BuildGPstart_time = time.time()
            GPs = []

            res = tmpR

            newX = np.delete(res, myfunc.input_dim, axis=1)
            newY = (np.delete(res, 0, axis=1)).tolist()
            for _ in range(myfunc.input_dim - 2):
                newY = (np.delete(newY, 0, axis=1)).tolist()
            newY = ((np.delete(newY, 0, axis=1)).reshape(1, -1)).tolist()

            n_in_X = (newX.tolist()).copy()
            n_in_Y = newY.copy()

            train_x = torch.tensor(newX, dtype=torch.float32).cuda()
            train_y = torch.tensor(newY, dtype=torch.float32).cuda()
            myGP = ExactGPModel(train_x, train_y).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)

            BuildGPstop_time = time.time()
            print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
        elif method == "KNN":
            R_in = np.append(n_in_X, n_in_Y, axis=1)
            RforIdx = R_in.copy()
            RforIdx[np.isnan(RforIdx)] = -1

            idxs = np.where(RforIdx == -1)

            tmpR = RforIdx.copy()
            imputer = KNNImputer(missing_values=-1, n_neighbors=3, weights="uniform")
            knnR = imputer.fit_transform(tmpR)

            for iii in range(len(idxs[0])):
                tmpR[idxs[0][iii], idxs[1][iii]] = knnR[idxs[0][iii], idxs[1][iii]]

            # Use the new predicted matrix as input training data for the GP
            BuildGPstart_time = time.time()
            GPs = []

            res = tmpR

            newX = np.delete(res, myfunc.input_dim, axis=1)
            newY = (np.delete(res, 0, axis=1)).tolist()
            for _ in range(myfunc.input_dim - 2):
                newY = (np.delete(newY, 0, axis=1)).tolist()
            newY = ((np.delete(newY, 0, axis=1)).reshape(1, -1)).tolist()

            n_in_X = (newX.tolist()).copy()
            n_in_Y = newY.copy()

            train_x = torch.tensor(newX, dtype=torch.float32).cuda()
            train_y = torch.tensor(newY, dtype=torch.float32).cuda()
            myGP = ExactGPModel(train_x, train_y).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)

            BuildGPstop_time = time.time()
            print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
        elif method == "uGP":
            if input_function == "HeatTreatment" or input_function == "RobotSim":
                for idxPoint in range(len(n_in_X)):
                    for idx2Point in range(idxPoint+1, len(n_in_X)):
                        checkdis = np.linalg.norm(np.array(n_in_X[idxPoint]) - np.array(n_in_X[idx2Point]),2)
                        if checkdis < 1.1:
                            if n_in_Y[idxPoint] >= n_in_Y[idx2Point]:
                                n_in_X.remove(n_in_X[idx2Point])
                                in_X.remove(in_X[idx2Point])
                                n_in_Y.remove(n_in_Y[idx2Point])
                                in_Y.remove(in_Y[idx2Point])
                            else:
                                n_in_X.remove(n_in_X[idxPoint])
                                n_in_Y.remove(n_in_Y[idxPoint])
                                in_X.remove(in_X[idxPoint])
                                in_Y.remove(in_Y[idxPoint])
                            break

            # instantiate and fit the KDE model
            kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
            kde.fit(n_in_X)

            # score_samples returns the log of the probability density
            logprobs = kde.score_samples(n_in_X)

            # Use the new predicted matrix as input training data for the GP
            BuildGPstart_time = time.time()
            GPs = []

            newX = (logprobs.tolist()).copy()
            newY = n_in_Y.copy()

            train_x = torch.tensor(newX, dtype=torch.float32).cuda()
            train_y = torch.tensor(newY, dtype=torch.float32).cuda()
            myGP = ExactGPModel(train_x, train_y).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)
        else:
            train_x = torch.tensor(n_in_X, dtype=torch.float32).cuda()
            train_y = torch.tensor(n_in_Y, dtype=torch.float32).cuda()
            myBO.optimizeHyperparamaters(myGP, train_x, train_y, GPHypers_optim, GPoptimIter)

        log_iter += str(np.max(np.union1d(in_Y, Yori))) + '\n'
        # Strategy for choosing the next points
        if method == "Suggest":
            testX = [myfunc.randUniformInNBounds() for i in range(10000)]
            testX = torch.tensor(testX).cuda()
            # Set into posterior mode
            myGP.eval()
            myGP.likelihood.eval()

            #### TS ####
            # preds = myGP.likelihood(myGP(testX))
            # newSample = preds.sample()
            # newSample = newSample.cpu()
            # nextX = testX[np.argmax(newSample)]
            # nextX = nextX.detach().cpu().numpy()

            #### GPUCB ####
            acq, _, _ = myBO.gpucb(testX, myGP, BetaT)
            # acqs.append(acq.tolist())
            nextX = testX[np.argmax(acq)]
            nextX = nextX.detach().cpu().numpy()
        elif method == "BOMI":
            opt_sug_starttime = time.time()
            # Sample the next point from each GP
            testX = [myfunc.randUniformInNBounds() for i in range(10000)]

            candidatesX = testX.copy()
            testX = torch.tensor(testX).cuda()
            nextXs = []
            nextXsAcq = []

            acqs = []
            for ii in range(numGPs):
                GPs[ii].eval()
                GPs[ii].likelihood.eval()
                acq, mean_i, var_i = myBO.gpucb(testX, GPs[ii], BetaT)
                acqs.append(acq)

            sAcq = np.mean(acqs, axis=0) + BetaT*np.std(acqs, axis=0)

            tmpX = testX[np.argmax(sAcq)]
            tmpX = tmpX.detach().cpu().numpy()
            nextXs.append(tmpX.tolist())
            nextXsAcq.append(sAcq[np.argmax(sAcq)])

            nextX = nextXs[np.argmax(nextXsAcq)]

            opt_sug_stoptime = time.time()
            print("nextX:", nextX, " idx:", np.argmax(nextXsAcq))
        elif method == "Mean":
            ######################### Random sampling #########################
            testX = [myfunc.randUniformInNBounds() for i in range(10000)]
            testX = torch.tensor(testX).cuda()

            acq, _, _ = myBO.gpucb(testX, myGP, BetaT)

            nextX = testX[np.argmax(acq)]
            nextX = nextX.detach().cpu().numpy()
        elif method == "uGP":
            testX = [myfunc.randUniformInNBounds() for i in range(10000)]
            logprobsX = kde.score_samples(testX)
            probX = torch.tensor(logprobsX).cuda()

            acq, _, _ = myBO.gpucb(probX, myGP, BetaT)

            nextX = testX[np.argmax(acq)]
            print("nextX:",nextX)
        else:
            ######################### Random sampling #########################
            testX = [myfunc.randUniformInNBounds() for i in range(10000)]
            testX = torch.tensor(testX).cuda()

            acq, _, _ = myBO.gpucb(testX, myGP, BetaT)

            nextX = testX[np.argmax(acq)]
            nextX = nextX.detach().cpu().numpy()

        # Query true objective function
        nextY, out_X = myfunc.func_with_missing(myfunc.denormalize(nextX))
        print("Out X:", out_X)

        # Augument data and update the statistical model
        if method == "Drop":
            if np.isnan(out_X).any():
                # We skip the observation if there is a missing value
                print("Drop!!")
                continue
            else:
                # Add to D (DropBO method)
                in_X.append(myfunc.denormalize(nextX))
                n_in_X.append(nextX)
                in_Y.append(nextY)
        elif method == "Suggest" or method == "uGP":
            in_X.append(myfunc.denormalize(nextX))
            n_in_X.append(nextX)
            in_Y.append(nextY)
        else:
            # Add to D
            nout_X = myfunc.normalize(out_X)
            in_X.append(myfunc.denormalize(nout_X))
            n_in_X.append(nout_X)
            in_Y.append(nextY)

        print("Next X:", myfunc.denormalize(nextX), " nextY:", nextY)
        minY = np.min(in_Y)
        maxY = np.max(in_Y)
        if method == "BPMF":
            n_in_Y = [[(((nb - na) * (tmpY - minY) / (maxY - minY) + na))] for tmpY in in_Y]
        elif method == "BOMI" or method == "Mean" or method == "Mode" or method == "KNN":
            n_in_Y = [[(((nb - na) * (tmpY - minY) / (maxY - minY) + na))] for tmpY in in_Y]
            in_Guessed_nY = [(((nb - na) * (tmpY - np.min(in_Guessed_Y)) / (np.max(in_Guessed_Y) - np.min(in_Guessed_Y)) + na)) for tmpY in in_Guessed_Y]
        else:
            if method == "Suggest" or method == "Drop" or method == "uGP":
                if minY == maxY:
                    minY -= 0.01

            n_in_Y = [(((nb - na) * (tmpY - minY) / (maxY - minY) + na)) for tmpY in in_Y]

            train_x = torch.tensor(n_in_X, dtype=torch.float32)
            train_y = torch.tensor(n_in_Y, dtype=torch.float32)

            myGP = ExactGPModel(train_x, train_y)
        print("Iter ", ite, " Optimum Y: \033[1m", np.max(np.union1d(in_Y, Yori)), "\033[0;0m at: ",
              myfunc.denormalize(n_in_X[np.argmax(in_Y)]))

    BOstop_time = time.time()
    ymax = np.max(np.union1d(in_Y, Yori))
    print("Run:", run, " method: ", method, " y: ", str(ymax), " numCalls: ", myfunc.numCalls, " ite:", ite,
          " time: --- %s seconds ---" % (BOstop_time - BOstart_time))

    ####################### END BO loops #######################

print("Solution: x=", in_X[np.argmax(in_Y)], " f(x)=", np.max(in_Y))