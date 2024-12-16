# Imports
import numpy as np
from sklearn.metrics import accuracy_score


def generateNoisySamples(
    samples, noiseFactor=0.01, noiseType='arithmetic_std', noiseScope='feature', noiseDistribution='fixed'
):
    '''
    Parameters
    ----------
    samples : array_like
        The samples that the noise will be applied to.
        The shape of samples define the shape of the returned array.
        shape (m,n) with m samples with dimension n.
    noisefactor : float
        Denotes the scale of noise to be added to the sample.
    noiseType : str
        Can take values of 'aritmetic_std', 'arithmetic' or 'geometric'.
        arithmetic_std - forces the noiseScope to be the standard deviation for each feature in the set of samples.
        The noise is added to the existing samples.;
        arithmetic - allows a variety of scopes to be selected. The noise is added to the existing samples.;
        geometric - ignores noiseScope. The noise is added via multiplication.
    noiseScope : str
        Can take values of 'local', 'sample', 'feature' or 'global':
        local - takes the absolute value for every element of every sample;
        sample - takes the highest absolute element value from each sample;
        feature - takes the highest absolute value for each feature in the set of samples;
        global - takes the highest absolute value of all elements in the set of samples.
        Only used when noiseType=='arithmetic'.
    noiseDistribution : str
        Can take values of 'fixed' or 'normal'.

    Returns
    -------
    numpy ndarray
        array of shape (m,n) comprising m samples with dimension n.
    '''

    # noise initialisation
    # we always initialise using normal noise, and then convert to 'fixed' noise if needed
    randomNoise = np.random.normal(size=samples.shape)

    if noiseDistribution == 'fixed':
        randomNoise = np.where(randomNoise>0,1,-1)

    if noiseType == 'arithmetic_std':
        scopeFactor = np.std(samples,axis=0).reshape((1,-1))
        noisySamples = samples + (noiseFactor * scopeFactor * randomNoise)

    elif noiseType == 'arithmetic':
        if noiseScope == 'global':
            scopeFactor = np.max(np.abs(samples)) # take the highest absolute value of all elements in the set of samples
        elif noiseScope == 'feature':
            scopeFactor = np.max(np.abs(samples),axis=0).reshape((1,-1)) # take the highest absolute value for each feature in the set of samples
        elif noiseScope == 'sample':
            scopeFactor = np.max(np.abs(samples),axis=1).reshape((-1,1)) # take the highest absolute element value from each sample
        elif noiseScope == 'local':
            scopeFactor = np.abs(samples) # take the absolute value for every element of every sample
        else:
            raise ValueError('Incorrect noiseType selected. Please select one of \'local\', \'sample\', \'feature\', or \'global\'.')

        noisySamples = samples + (noiseFactor * scopeFactor * randomNoise)

    elif noiseType == 'geometric':
        noisySamples = samples * (1 + noiseFactor * randomNoise)

    else:
        raise ValueError('Incorrect noiseType passed to generateNoisySamples function.')

    return noisySamples


def generateNoisyAccuracyList(samples, labels, model, noiseFactor=0.01, noiseType='arithmetic', \
                          noiseScope='feature', noiseDistribution='fixed', numberOfNoiseIterations=100):

        '''
        Parameters
        ----------
        the function takes the same parameters as 'generateNoisySamples'
        in addition, the function takes a further parameter:
        numberOfNoiseIterations : integer
            this defines the number of times noise will be sampled for the dataset that is passed to the function.

        Returns
        -------
        list
            a list of the accuracy figures computed over the model for each iteration of noise sampling.
        '''

        noisyAccuracyList = []

        for iteration in range(numberOfNoiseIterations):

            noisySamples = generateNoisySamples(samples=samples, noiseFactor=noiseFactor, noiseType=noiseType, \
                                                noiseScope=noiseScope, noiseDistribution=noiseDistribution)
            noisyPredictions = model.predict(noisySamples)
            noisyAccuracy = accuracy_score(noisyPredictions, labels)
            noisyAccuracyList.append(noisyAccuracy)

        return noisyAccuracyList


