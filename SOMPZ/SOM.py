import sys
from NoiseSOM import NoiseSOM, hFunc, AsinhMetric
import numpy as np
import scipy
import os, sys, time
import multiprocessing as mp
import warnings


def TrainSOM(flux, flux_err, Ncells = 48, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  lnScaleStep = 0.03, minError = 0.02):

    nTrain = len(flux)

    hh      = hFunc(nTrain, sigma = hFunc_sigma)
    metric  = AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)

    som = NoiseSOM(metric, flux, flux_err, learning=hh, shape=(Ncells, Ncells), 
                   wrap=False, logF=True, initialize='sample', minError = minError)

    # And return the resultant weight matrix    
    return som.weights    


def ClassifySOM(flux, flux_err, som_weights, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  lnScaleStep = 0.03, minError = 0.02):

    nTrain = len(flux)

    hh      = hFunc(nTrain, sigma = hFunc_sigma)
    metric  = AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)

    som = NoiseSOM(metric, None, None, learning=hh, shape = som_weights.shape[:-1], 
                   wrap=False, logF=True, initialize=som_weights, minError = minError)

    return som.classify(flux, flux_err)


def Classifier(som_weights, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  lnScaleStep = 0.03, minError = 0.02):

    metric  = AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)
    som     = NoiseSOM(metric, None, None, learning = None, shape = som_weights.shape[:-1], 
                       wrap=False, logF=True, initialize=som_weights, minError = minError)

    return som