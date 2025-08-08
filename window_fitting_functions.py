import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import scipy.optimize



def return_W_index(channel_index, cw_subchannel, sub_channels_per_channel=10):
    return int(np.abs(cw_subchannel - channel_index)*sub_channels_per_channel)

def window_fitting_func(spectrum, w_abs_squared, n_fitting_channels):
    
    channel_with_cw = np.argmax(spectrum)
    cw_subchannel_lim = [channel_with_cw-0.5, channel_with_cw+0.5]

    sub_channels_per_channel = int(len(w_abs_squared) / len(spectrum))
    sub_channel_resolution = 1 / sub_channels_per_channel

    fitting_channels = spectrum[channel_with_cw-n_fitting_channels:channel_with_cw+n_fitting_channels]
    fitting_indices = np.arange(start=channel_with_cw-n_fitting_channels, stop=channel_with_cw+n_fitting_channels)

    cw_subchannel_list = np.arange(start=cw_subchannel_lim[0],
                                   stop=cw_subchannel_lim[-1]+sub_channel_resolution,
                                   step=sub_channel_resolution)
    
    prior_n_p_mean = np.mean(spectrum[channel_with_cw+n_fitting_channels+1:channel_with_cw+n_fitting_channels+10])
    prior_n_p_span = 5 * np.std(spectrum[channel_with_cw+n_fitting_channels+1:channel_with_cw+n_fitting_channels+10])
    if prior_n_p_mean - prior_n_p_span <= 0:
        np_lb = 0
        np_ub = prior_n_p_mean + prior_n_p_span
    else:
        np_lb = prior_n_p_mean - prior_n_p_span
        np_ub = prior_n_p_mean + prior_n_p_span


    channel_limit_response = w_abs_squared[int(sub_channels_per_channel/2)] / w_abs_squared[0]

    p_cw_min = np.max(spectrum) - 2 * prior_n_p_mean
    p_cw_max = (np.max(spectrum) - prior_n_p_mean) / channel_limit_response
    p_cw_0 = np.mean(np.array([p_cw_min, p_cw_max]))

    x0 = [p_cw_0, prior_n_p_mean]# starting parameters for the minimisation



    #bounds = scipy.optimize.Bounds(lb=[0, 0],
    #                               ub=[np.inf, np.inf]) # set boundaries on the fitting parameters

    FoMs = []       # list of FoMs
    results = []    # list of resulst

    for cw_subchannel in cw_subchannel_list: # go through each CW subchannel and fit for P_cw and P_noise
        w_abs_indices = np.array([return_W_index(i, cw_subchannel, sub_channels_per_channel) for i in fitting_indices])
        w_abs_values = w_abs_squared[w_abs_indices]

        model_power = lambda x : (x[0]*w_abs_values)+x[1]

        fom_function = lambda x: np.sum(np.abs(fitting_channels - model_power(x)) / fitting_channels)

        res = scipy.optimize.minimize(fom_function, x0=x0) # minimise the fractional error to not bias the fitting
        
        FoMs.append(res.fun)
        results.append(res.x)

    FoMs = np.array(FoMs)

    optimum_results = results[np.argmin(FoMs)] # [P_cw, P_noise]
    optimum_cw_subchannel = cw_subchannel_list[np.argmin(FoMs)] # get cw sub channel with optimal FoM

    p_cw, p_noise = optimum_results

    optimum_cw_subchannel = round(optimum_cw_subchannel, int(np.log10(sub_channels_per_channel))+2)

    # calculate the CW power per channel using the difference between the channel indicies

    cw_leaked_powers = np.array([p_cw * w_abs_squared[return_W_index(index, optimum_cw_subchannel, sub_channels_per_channel)] for index in np.arange(len(spectrum))]) # check this

    cal_spectra = spectrum - cw_leaked_powers

    return (cal_spectra, [p_cw, p_noise, optimum_cw_subchannel])

def window_cw_fitting_for_array(array, window_function, w_abs_squared=None, sub_channels_per_channel=10, n_fitting_channels = 4, plot=False):
    if window_function is not None:
        w = window_function(len(array[0])) # produce window
        w_abs_squared = np.abs(np.fft.fft(w, n=len(w)*sub_channels_per_channel))**2
    else:
        if len(w_abs_squared) != len(array[0]) * sub_channels_per_channel:
            print('w_abs_squared is not the correct shape')
            print('w_abs_squared.shape = ', w_abs_squared.shape)
            print('fft_length * subchannels per channel = ', len(array[0]) * sub_channels_per_channel)
        
    array = [s for s in array]
    args = zip(array, itertools.repeat(w_abs_squared), itertools.repeat(n_fitting_channels))

    with multiprocessing.Pool() as pool:
        results = pool.starmap(window_fitting_func, args)

    cw_subtracted_spectra = np.array([s for s,_ in results])
    fitted_values = np.array([f for _,f in results])
    del results
    p_cw = fitted_values[:,0]
    p_n = fitted_values[:,1]
    cw_subchannels = fitted_values[:,2]

    if plot:
        plt.plot(p_cw)
        plt.show()

    return cw_subtracted_spectra, p_cw, p_n, cw_subchannels