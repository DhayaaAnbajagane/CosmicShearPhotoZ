import sys
from . import NoiseSOM as ns
import numpy as np
import scipy
import os, sys, time
import multiprocessing as mp
import warnings


def TrainSOM(flux, flux_err, Ncells = 48, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  lnScaleStep = 0.03, minError = 0.02):

    nTrain = len(flux)

    hh      = ns.hFunc(nTrain, sigma = hFunc_sigma)
    metric  = ns.AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)

    som = ns.NoiseSOM(metric, flux, flux_err, learning=hh, shape=(Ncells, Ncells), 
                      wrap=False, logF=True, initialize='sample', minError = minError)

    # And return the resultant weight matrix    
    return som.weights    


def ClassifySOM(flux, flux_err, som_weights, rng, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  lnScaleStep = 0.03, minError = 0.02):

    nTrain = len(flux)

    hh      = ns.hFunc(nTrain, sigma = hFunc_sigma)
    metric  = ns.AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)

    som = ns.NoiseSOM(metric, None, None, learning=hh, shape = som_weights.shape, 
                      wrap=False, logF=True, initialize=som_weights, minError = minError)

    return som.classify(flux, flux_err)