import numpy as np
import matplotlib.pyplot as plt


def linear(x, params):
    return x*params[0] + params[1]

class CW_Calibrator:
    def __init__(self, spectra, times, frequencies, pm_powers=None):
        self.spectra = spectra
        self.times = times
        self.frequencies = frequencies
        if pm_powers is None:
            self.pm_present = False
        pass

    def include_power_meter(self, power_meter_measurements, pm_power_func=linear, pm_params=[1,1], pm_zero_measurement=1):
        self.pm_present = True
        pm_zero_pwr = pm_power_func(pm_zero_measurement, pm_params)
        pm_pwrs = pm_power_func(power_meter_measurements, pm_params)

        self.pm_y = (pm_pwrs - pm_zero_pwr) / (pm_pwrs[0] - pm_zero_pwr)


    def aperture_cw_isolation(self, s, aperture_bounds, return_cw_index=False):
        n_cw_side_channels, n_non_cw_channels = aperture_bounds
        cw_index = np.argmax(s)
        non_cw_lower_mean = np.mean(s[cw_index-1-n_non_cw_channels-n_cw_side_channels : cw_index-1-n_cw_side_channels])
        non_cw_upper_mean = np.mean(s[cw_index+1+n_cw_side_channels : cw_index+1+n_non_cw_channels+n_cw_side_channels:])
        non_cw_mean = (non_cw_lower_mean + non_cw_upper_mean) / 2

        cw_channels = s[cw_index-n_cw_side_channels:cw_index+n_cw_side_channels]
        cw_pwr = np.sum(cw_channels) - non_cw_mean * len(cw_channels)
        if return_cw_index:
            return cw_pwr, cw_index
        else:
            return cw_pwr
        

    def polynomial_fit_cw_isolation(self, s, polynomial_params, return_cw_index=False):
        order, n_cw_side_channels, n_fitting_channels = polynomial_params
        cw_index = np.argmax(s)
        non_cw_channels_lower = np.arange(start=cw_index - n_cw_side_channels-1-n_fitting_channels, 
                                          stop=cw_index - n_cw_side_channels-1)
        
        non_cw_channels_upper = np.arange(start=cw_index + n_cw_side_channels+1, 
                                          stop=cw_index + n_cw_side_channels+1+n_fitting_channels)
        
        non_cw_indices = np.concatenate((non_cw_channels_lower, non_cw_channels_upper), axis=None)

        non_cw_pwrs = s[non_cw_indices]
        cw_indices = np.arange(start=cw_index - n_cw_side_channels,
                               stop=cw_index + n_cw_side_channels)        

        poly_fitted = np.poly1d(np.polyfit(non_cw_indices, non_cw_pwrs, deg=order))

        cw_ch_powers = s[cw_indices] - poly_fitted(cw_indices)

        cw_pwr = np.sum(cw_ch_powers)

        if return_cw_index:
            return cw_pwr, cw_index
        else:
            return cw_pwr
    


    def otf_gain_calibration(self, method='Aperture', aperture_bounds=[2, 3], polynomial_params=[3, 3, 20], return_measured_gains = False):
        """
        performs an on the fly gain calibration using the CW signal as a reference as well as power-meter
        measurement if available.

        method - 'Aperture', 'Polynomial' - choice of CW isolation
        aperture_bounds - [n_cw_sidechannels, n_non_cw_channels]
        polynomail_params - [fitting_order, n_cw_side_channels, n_fitting_channels]
        """
        if method == 'Aperture' or method == 'Ap':
            delta_p_cw = np.array([self.aperture_cw_isolation(s, aperture_bounds) for s in self.spectra])
        elif method == 'Polynomial' or method == 'Poly':
            delta_p_cw = np.array([self.polynomial_fit_cw_isolation(s, polynomial_params) for s in self.spectra])

        if self.pm_present:
            gains = delta_p_cw / (delta_p_cw[0] * self.pm_y)
        else:
            gains =  delta_p_cw / delta_p_cw[0]

        self.gain_corrected_spectras = self.spectra.T / gains
        self.gain_corrected_spectras = self.gain_corrected_spectras.T

        self.measured_gains = gains
        if return_measured_gains:
            return self.gain_corrected_spectras, self.measured_gains
        else:
            return self.gain_corrected_spectras
        
    
    def compare_pink_noise_before_and_after(self, n_to_mask=2, plot=False):
        cw_indices = [self.aperture_cw_isolation(s, return_cw_index=True, aperture_bounds=[1,1])[1] for s in self.spectra]
        mask = np.ones(shape=self.gain_corrected_spectras.shape) # create a range of values for the mask based on the min and max of the cw indices, collapse into just frequency

        min_max_cw_indices = [int(np.min(cw_indices)) - n_to_mask, int(np.max(cw_indices)) + n_to_mask]
        mask_indices = np.arange(len(self.frequencies))[min_max_cw_indices[0]:min_max_cw_indices[1]]
        
        gain_corrected_psd_list = []
        for nu, tod in enumerate(self.gain_corrected_spectras.T):
            if nu in mask_indices:
                pass
            else:
                psd = np.abs(np.fft.rfft(tod/np.mean(tod)))
                gain_corrected_psd_list.append(psd)

        gain_corrected_psd = np.mean(np.array(gain_corrected_psd_list), axis=0)

        original_psd_list = []
        for nu, tod in enumerate(self.spectra.T):
            if nu in mask_indices:
                pass
            else:
                psd = np.abs(np.fft.rfft(tod / np.mean(tod)))
                original_psd_list.append(psd)
        original_psd = np.mean(np.array(original_psd_list), axis=0)
        
        fft_freqs = np.fft.rfftfreq(n=len(self.spectra.T[0]), d=self.times[-1] / len(self.times))
        if plot:
            plt.plot(fft_freqs[1:], original_psd[1:], label='Original')
            plt.plot(fft_freqs[1:], gain_corrected_psd[1:], label='Gain Corrected')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('PSD')
            plt.yscale('log')
            plt.xscale('log')
            plt.tight_layout()
            plt.show()
        return [original_psd, gain_corrected_psd, fft_freqs]


class AbsoluteCalibrator:
    def __init__(self, spectra, switch_list, switch_dict, load_estimate, ns_estimate):
        self.initial_spectra = spectra
        self.switch_list = switch_list
        self.switch_dict = switch_dict
        self.load_estimate = load_estimate
        self.ns_esitimate = ns_estimate

        ## split the spectra into the various targets

        pass

    def rough_absolute_calibration(self):
        
        pass