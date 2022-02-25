import scipy
import numpy as np

from scipy.stats import norm
from scipy import interpolate

class Norm_Sc(object):
    def __init__(self, data, frequency, max_sc, min_sc, power):
        # Input distribution that needs to be transformed before the inversion
        self.data = np.array(data)
        self.frequency = np.array(frequency)

        # Maximum and minimum allowable scores
        self.max_sc = max_sc
        self.min_sc = min_sc

        # Power for interpolating the normal scores that outside the range of input distribution
        self.power = power

        # The normal score for the transformation function
        self.score = None

    def score_calc(self):
        # Sorting the distribution before the inversion
        self.data = self.data[np.argsort(self.data)]
        self.frequency = self.frequency[np.argsort(self.data)]

        # Normalizing and averaging the input cumulative distribution
        cum_norm_frequency_a = np.cumsum(self.frequency/np.sum(self.frequency))
        cum_norm_frequency_b = np.append(np.array([0]), cum_norm_frequency_a[:-1])
        ave_CNF = 0.5*(cum_norm_frequency_a + cum_norm_frequency_b)

        # Obtain the corresponding normal score for the input cumulative distribution
        self.score = np.array([norm.ppf(f) for f in ave_CNF])

    def norm_score_trans(self, da):
        self.score_calc()
        # Transform the input distribution to Gaussian before the inversion)
        trans_func = interpolate.interp1d(self.data,self.score,fill_value="extrapolate")

        return trans_func(da)

    def reverse_norm_score_trans(self, sc):
        self.score_calc()
        # Array storing the reversely transformed data from Gaussian to the input distribution
        new_data = np.empty(np.shape(sc))
        new_data[:] = np.nan

        # Getting the minimum and maximum data and score before the inversion
        min_data = self.data[0]
        max_data = self.data[-1]
        min_score = self.score[0]
        max_score = self.score[-1]

        # Getting the normal scores that fall within the range of input distribution
        mid_score_idx = np.logical_and(sc <= max_score, sc >= min_score)
        reverse_trans_func = interpolate.interp1d(self.score,self.data,fill_value="extrapolate")
        new_data[mid_score_idx] = reverse_trans_func(sc[mid_score_idx])

        # Interpolates the scores that fall below the range of input distribution
        blw_score_idx = sc < min_score
        s = list()
        for scb in sc[blw_score_idx]:
            cdflo = norm.cdf(min_score)
            cdfbl = norm.cdf(scb)
            if self.power != 0:
                # Interpolate the scores given the corresponding cdf
                s.append(self.scint(0, cdflo, self.min_sc, min_data, cdfbl, self.power))
            else:
                s.append(min_data)

        new_data[blw_score_idx] = s

        # Interpolates the scores that fall above the range of input distribution
        abv_score_idx = sc > max_score
        s = list()
        for sca in sc[abv_score_idx]:
            cdfup = norm.cdf(max_score)
            cdfab = norm.cdf(sca)
            if self.power != 0:
                # Interpolate the scores given the corresponding cdf
                s.append(self.scint(cdfup, 1.0, max_data, self.max_sc, cdfab, self.power))
            else:
                s.append(max_data)

        new_data[abv_score_idx] = s

        return new_data

    # Interpolates the normal scores that outside the range of input distribution
    def scint(self, cdflow, cdfhigh, sclow, schigh, targetcdf, power):
        # cdf can be thought as the x-variable here and score (sc) can be thought as the y-variable here
        # targetcdf is the cdf of the score we want to reverse
        if cdfhigh-cdflow < np.finfo(float).eps:
            return (schigh + sclow) / 2.0
        else:
            return sclow + (targetcdf - cdflow)*(((schigh - sclow)/(cdfhigh - cdflow))**power)
