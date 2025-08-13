import numpy as np
import h5py
import window_fitting_functions
import os
import gains_analysis
import scipy.signal
from charger import *


CW_AMPS = [0.01, 0.1,1,10, 100]
SUB_CHANNELS_PER_CHANNEL_LIST = [10,100,1000,10000]
SAVE_PATH = 'Test_Results/Window_Tests'

Bartlett_string_list = ['Generated_Data/WindowTests_LaptopRan/winBartlett_cwamp0.01_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBartlett_cwamp0.1_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBartlett_cwamp1_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBartlett_cwamp10_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBartlett_cwamp100_test.hd5f']

Blackman_string_list = ['Generated_Data/WindowTests_LaptopRan/winBlackman_cwamp0.01_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBlackman_cwamp0.1_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBlackman_cwamp1_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBlackman_cwamp10_test.hd5f',
                        'Generated_Data/WindowTests_LaptopRan/winBlackman_cwamp100_test.hd5f']

BlackmanHarris_string_list = ['Generated_Data/WindowTests_LaptopRan/winBlackmanHarris_cwamp0.01_test.hd5f',
                              'Generated_Data/WindowTests_LaptopRan/winBlackmanHarris_cwamp0.1_test.hd5f',
                              'Generated_Data/WindowTests_LaptopRan/winBlackmanHarris_cwamp1_test.hd5f',
                              'Generated_Data/WindowTests_LaptopRan/winBlackmanHarris_cwamp10_test.hd5f',
                              'Generated_Data/WindowTests_LaptopRan/winBlackmanHarris_cwamp100_test.hd5f']


Cosine_string_list  = ['Generated_Data/WindowTests_LaptopRan/winCosine_cwamp0.01_test.hd5f',
                       'Generated_Data/WindowTests_LaptopRan/winCosine_cwamp0.1_test.hd5f',
                       'Generated_Data/WindowTests_LaptopRan/winCosine_cwamp1_test.hd5f',
                       'Generated_Data/WindowTests_LaptopRan/winCosine_cwamp10_test.hd5f',
                       'Generated_Data/WindowTests_LaptopRan/winCosine_cwamp100_test.hd5f']

Rectanuglar_string_list = ['Generated_Data/WindowTests_LaptopRan/winRectangular_cwamp0.01_test.hd5f',
                           'Generated_Data/WindowTests_LaptopRan/winRectangular_cwamp0.1_test.hd5f',
                           'Generated_Data/WindowTests_LaptopRan/winRectangular_cwamp1_test.hd5f',
                           'Generated_Data/WindowTests_LaptopRan/winRectangular_cwamp10_test.hd5f',
                           'Generated_Data/WindowTests_LaptopRan/winRectangular_cwamp100_test.hd5f']



def apply_gain_correction(spectra, p_cw, p_pm=None):
    corrected_spectra = []

    if p_pm is not None:
        gains = (p_cw * p_pm[0]) / (p_cw[0] * p_pm)
    else:
        gains = p_cw / p_cw[0]
    
    corrected_spectra = []
    for i, s in enumerate(spectra):
        corrected_spectra.append(s/gains[i])
    corrected_spectra = np.array(corrected_spectra)
    return corrected_spectra

def compute_PSD(spectra, times, n_to_mask=3):
    cw_indices = [np.argmax(s) for s in spectra]

    min_max_cw_indices = [int(np.min(cw_indices)) - n_to_mask, int(np.max(cw_indices)) + n_to_mask] # create slices to exclude
    mask_indices = np.arange(len(spectra[0]))[min_max_cw_indices[0]:min_max_cw_indices[1]] # indices to mask

    psd_list = []
    for nu, tod in enumerate(spectra.T):
        if nu in mask_indices: # skip channels that are masked
            pass
        else:
            psd = np.abs(np.fft.rfft(tod/np.mean(tod)))
            psd_list.append(psd)

    psd = np.mean(np.array(psd_list), axis=0)
    fft_freqs = np.fft.rfftfreq(n=len(spectra.T[0]), d=times[-1] / len(times))
    return psd, fft_freqs



def process_and_save_tests(charger_obs_string_list,
                           window_function,
                           window_function_string,
                           cw_amps=CW_AMPS,
                           sub_channels_per_channels_list=SUB_CHANNELS_PER_CHANNEL_LIST,
                           save_path=SAVE_PATH):

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = save_path+'/'

    with h5py.File(name=f'{save_path}/{window_function_string}.hd5f') as file:
        print(f'Processing {window_function_string}')
        for filename, cw_amp in zip(charger_obs_string_list, cw_amps):
            cw_amp_grp = file.create_group(f'CwAmp_{cw_amp}')
            obs = TimeStreamGenerator()
            obs.read_in_from_h5py(filename)
            original_psd, psd_freqs = compute_PSD(obs.integrated_spectra, obs.times, n_to_mask=10)
            cw_amp_grp.create_dataset('Original_PSD', data=original_psd, dtype=original_psd.dtype)
            cw_amp_grp.create_dataset('PSD_Freqs', data=psd_freqs, dtype=psd_freqs.dtype)

            aperture_method = gains_analysis.CW_Calibrator(obs.integrated_spectra, obs.times, obs.frequencies_mhz)
            aperture_spectra, cw_gains = aperture_method.otf_gain_calibration(method='Aperture', aperture_bounds=[3,3], return_measured_gains=True) # gain calibrated
            scs_method_psd,_ = compute_PSD(aperture_spectra, obs.times, n_to_mask=10)
            cw_amp_grp.create_dataset('SCS_PSD', data=scs_method_psd, dtype=scs_method_psd.dtype)

            window_grps = cw_amp_grp.create_group('WindowFitGroup')

            for subchannel_per_channel in sub_channels_per_channels_list: # compute the 
                print(f'Processing {window_function_string}')
                print(f'Processing| CW Amp: {cw_amp}, subChannelRes: {subchannel_per_channel}')

                sub_channel_grp = window_grps.create_group(f'SubChannel_{subchannel_per_channel}')
                cw_corrected_spectra, p_cw, p_n, cw_subchannels = window_fitting_functions.window_cw_fitting_for_array(obs.integrated_spectra,
                                                                                                                   window_function,
                                                                                                                   sub_channels_per_channel=subchannel_per_channel,
                                                                                                                   n_fitting_channels=8, plot=False)

                sub_channel_grp.create_dataset('CwCorrectedSpectra', data=cw_corrected_spectra, dtype=cw_corrected_spectra.dtype)
                sub_channel_grp.create_dataset('CwPowers', data=p_cw, dtype=p_cw.dtype)

                window_calibrated_spectra =  apply_gain_correction(spectra=cw_corrected_spectra, p_cw=p_cw)

                sub_channel_grp.create_dataset('GainCorrectedSpectra', data=window_calibrated_spectra, dtype=window_calibrated_spectra.dtype)

                window_method_psd,_ = compute_PSD(window_calibrated_spectra, obs.times, n_to_mask=10)

                sub_channel_grp.create_dataset('WindowMethod_PSD', data=window_method_psd, dtype=window_method_psd.dtype)

        
                pass

            pass

if __name__ == "__main__":
    process_and_save_tests(Blackman_string_list, np.blackman, 'Blackman')
    process_and_save_tests(Rectanuglar_string_list, signal.windows.boxcar, 'Rectangular')
    process_and_save_tests(Bartlett_string_list, np.bartlett, 'Bartlett')
    process_and_save_tests(BlackmanHarris_string_list, signal.windows.blackmanharris, 'BlackmanHarris')
    process_and_save_tests(Cosine_string_list, signal.windows.cosine, 'Cosine')

    print("@@@@@@@@@@@@@@@@")
    print('--- All Done ---')
    print("@@@@@@@@@@@@@@@@")