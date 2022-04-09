def Kalman_generator(Z, Yj, Y, R, invR, hk_update, error_count, min_da, max_da, power):
    try:
        # Z from the previous iteration
        #Z_prev = deepcopy(Z)

        # determine parameters from input
        num_parameters, Ne = Z.shape
        num_obs = Y.size

        # Normal score transformation
        for i in range(Ne):
            # For Z
            # zhist, zbins = np.histogram(Z[non_zero_idx,i],bins=nb)
            zhist, zbins = np.histogram(Z[:, i], bins=nb)
            NS = Norm_Sc()
            # zscore (x-axis of the new distribution)
            # Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx,i])
            Z[:, i], zscore[:, i] = NS.norm_score_trans(zbins[:-1], zhist, Z[:, i])
            # The original bins
            zbin[:, i] = zbins[:-1]

        # calculate ensemble means for upcoming covariance calculations
        Zbar = np.mean(Z, axis=1)
        Ybar = np.mean(Yj, axis=1)

        # Calculate covariance and parameter cross-covariance
        Pzy = np.zeros((num_parameters, num_obs))
        Pyy = np.zeros((num_obs, num_obs))

        # Calculate parameter covariance
        for i in range(Ne):
            sig_z = np.array([Z[:,i]-Zbar])
            sig_y = np.array([Yj[:,i]-Ybar])
            Pyy += np.dot(sig_y.T,sig_y)
            Pzy += np.dot(sig_z.T,sig_y)

        Pzy = Pzy/Ne
        Pyy = Pyy/Ne

        # Kalman gain matrix G
        G = np.dot(Pzy, np.linalg.inv(Pyy+R))

        Li = np.zeros((Ne))

        # Updated parameters
        for i in range(Ne):
            # calculate residuals
            ri = np.array([Y-Yj[:,i]])
            Li[i] = np.dot(np.dot(ri, invR), ri.T)
            Lif = Li[i]*2

            a = 0.99
            count = 0

            # Won't run when count is more than 11
            while Lif > Li[i] and count < 11:
                Z[:,i] = Z[:,i] + np.squeeze(a*np.dot(G, ri.T))

                # Reverse normal score transformation of z
                #nor, b = np.histogram(Z[non_zero_idx,i],bins=nb)
                nor, b = np.histogram(Z[:, i], bins=nb)
                NS = Norm_Sc()
                #Z[non_zero_idx,i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:,i], zbin[:,i], Z[non_zero_idx,i], min_da, max_da, power)
                Z[:, i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:, i], zbin[:, i], Z[:, i], min_da, max_da, power)

                # Get the updated hydraulic conductivity
                hk_update = np.exp(Z[:, i])
                #hk_update[zero_idx]= 0
                hk_update = np.reshape(hk_update, (nlay, nrow, ncol))

                # Rerun model with new parameters
                mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_update, prsity, al,
                                                                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

                at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

                # Forcasted state
                yf = at_ensem.flatten()
                # Updated parameters
                hk_update = hk_update.flatten()

                # calculate residuals
                ri = np.array([Y-yf])
                Lif = np.dot(np.dot(ri, invR), ri.T)
                Yj[:, i] = yf
                count += 1
                a = a/2

                # Triggered when the Lif is not reduced to smaller than Li[i]
                if count == 10:
                    print('Min dampening factor reached on realization: ' + str(i+1))

                    # If no acceptance could be reached, ensemble member i is resampled from the full ensemble of last iteration
                    #rand1 = np.random.choice([x for x in range(Ne-1)])
                    #rand2 = np.random.choice([x for x in range(Ne-1)])
                    #Z[non_zero_idx, i] = (Z_prev[non_zero_idx, rand1] + Z_prev[non_zero_idx, rand2]) / 2

                    # Normal score transformation of z for the next update
                    #zhist, zbins = np.histogram(Z[non_zero_idx,i],bins=nb)
                    zhist, zbins = np.histogram(Z[:, i], bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    #Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx,i])
                    Z[:, i], zscore[:, i] = NS.norm_score_trans(zbins[:-1], zhist, Z[:, i])
                    # The original bins
                    zbin[:,i] = zbins[:-1]
                else:
                    # Normal score transformation of z for the next update
                    # zhist, zbins = np.histogram(Z[non_zero_idx,i],bins=nb)
                    zhist, zbins = np.histogram(Z[:, i], bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    # Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx,i])
                    Z[:, i], zscore[:, i] = NS.norm_score_trans(zbins[:-1], zhist, Z[:, i])
                    # The original bins
                    zbin[:, i] = zbins[:-1]

            # Reverse normal score transformation of z
            # nor, b = np.histogram(Z[non_zero_idx,i],bins=nb)
            nor, b = np.histogram(Z[:, i], bins=nb)
            NS = Norm_Sc()
            # Z[non_zero_idx,i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:,i], zbin[:,i], Z[non_zero_idx,i], min_da, max_da, power)
            Z[:, i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:, i], zbin[:, i], Z[:, i], min_da, max_da, power)
    except Exception as e:
        error_count += 1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error message: ' + str(e) + '; line: ' + str(exc_tb.tb_lineno))
        np.savetxt(workdir + '\\hk_problem' + str(error_count) + '.csv', Z[:,i], delimiter=',')


    return Z, Yj, Li, error_count
