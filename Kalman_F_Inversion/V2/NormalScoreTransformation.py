import scipy
import numpy as np

from scipy.stats import norm
from scipy import interpolate

class Norm_Sc(object):
    # From original data to normal distribution
    def norm_score_trans(self, data, frequency, dist):
        # Sorting the distribution before the inversion
        sorted_idx = np.argsort(data)
        rev_sorted_idx = np.argsort(sorted_idx)

        # Sorting the index of the original distribution
        dist_idx = np.argsort(dist)
        rev_dist_idx = np.argsort(dist_idx)

        data = data[sorted_idx]
        frequency = frequency[sorted_idx]

        # Normalizing and averaging the input cumulative distribution
        cum_norm_frequency_a = np.cumsum(frequency/np.sum(frequency))
        cum_norm_frequency_b = np.append(np.array([0]), cum_norm_frequency_a[:-1])
        ave_CNF = 0.5*(cum_norm_frequency_a + cum_norm_frequency_b)

        # Obtain the corresponding normal score for the input cumulative distribution
        score = np.array([norm.ppf(f) for f in ave_CNF])

        # Transform the input distribution (bins) to Gaussian before the inversion
        trans_func = interpolate.interp1d(data, score)
        nbins = trans_func(data)

        # Generate the normal transformed array in its original indexing
        norm_trans = np.repeat(nbins, frequency)
        norm_trans = norm_trans[rev_dist_idx]

        #print(dist_idx)
        #print(rev_dist_idx)
        nbins = nbins[rev_sorted_idx]

        return norm_trans, nbins

    # From normal to original data distribution
    def reverse_norm_score_trans(self, sc, hist, scr, dat, inp_dist, min_da, max_da, power):
        # Sorting the index of the original distribution
        dist_idxr = np.argsort(inp_dist)
        rev_dist_idxr = np.argsort(dist_idxr)

        # Array storing the reversely transformed data from Gaussian to the input distribution
        nbinsr = np.empty(np.shape(sc))
        nbinsr[:] = np.nan

        # Getting the minimum and maximum data and score before the inversion
        min_data = dat[0]
        max_data = dat[-1]
        min_score = scr[0]
        max_score = scr[-1]

        # Getting the normal scores that fall within the range of input distribution
        mid_score_idx = np.logical_and(sc <= max_score, sc >= min_score)
        reverse_trans_func = interpolate.interp1d(scr,dat)
        inrange_sc = sc[mid_score_idx]
        nbinsr[mid_score_idx] = reverse_trans_func(inrange_sc)

        # Interpolates the scores that fall below the range of input distribution
        blw_score_idx = sc < min_score
        s = list()
        for scb in sc[blw_score_idx]:
            cdf_minsc = norm.cdf(min_score)
            cdf_below = norm.cdf(scb)
            cdf_floor = 0.001               # CDF of the min_dat
            # To prevent out of range data transformation (e.g. data with extreme CDF)
            if cdf_below < cdf_floor:
                cdf_below = cdf_floor

            # Interpolate the scores given the corresponding cdf
            s.append(self.scint(cdf_floor, cdf_minsc, min_da, min_data, cdf_below, power))

        nbinsr[blw_score_idx] = s

        # Interpolates the scores that fall above the range of input distribution
        abv_score_idx = sc > max_score
        s = list()
        for sca in sc[abv_score_idx]:
            cdf_maxsc = norm.cdf(max_score)
            cdf_above = norm.cdf(sca)
            cdf_ceil = 0.999               # CDF of the max_dat
            # To prevent out of range data transformation (e.g. data with extreme CDF)
            if cdf_above > cdf_ceil:
                cdf_above = cdf_ceil

            # Interpolate the scores given the corresponding cdf
            s.append(self.scint(cdf_maxsc, cdf_ceil, max_data, max_da, cdf_above, power))

        nbinsr[abv_score_idx] = s

        # Generate the normal transformed array in its original indexing
        rev_norm_trans = np.repeat(nbinsr, hist)
        rev_norm_trans = rev_norm_trans[rev_dist_idxr]

        return rev_norm_trans

    # Interpolates the normal scores that outside the range of input distribution
    # Takein x_want to get y_want
    def scint(self, xlow, xhigh, ylow, yhigh, x_want, power):
        # cdf can be thought as the x-variable here and score (sc) can be thought as the y-variable here
        # targetcdf is the cdf of the score we want to reverse
        # adjust_factor = np.finfo(float).eps
        adjust_factor = 10**-3
        if abs(xhigh-xlow) < adjust_factor:
            return (yhigh + ylow) / 2.0
        else:
            return ylow + (x_want - xlow)*(((yhigh - ylow)/(xhigh - xlow))**power)


