import numpy as np
import matplotlib.pyplot as plt
import h5py
import multiprocessing
import argparse
import itertools

import scipy.fft

def F(f, f_k, alpha, delta_nu):
    return ((f_k / f)**alpha) / delta_nu

def PSD_F(f, omega, f_k, alpha, omega_o, beta, delta_nu, n_channels, sample_rate):
    return F(f, f_k, alpha, delta_nu) #* H(omega, omega_o, beta) * C(beta, n_channels, sample_rate, omega_o)

def generate_F_psd(f_k=1, alpha=2, receiver_sample_rate=20e6, n_times=1024, tau=1, n_channels=1024):
    delta_nu = receiver_sample_rate / n_channels
    fourier_freqs = np.fft.rfftfreq(n_times+2, d=tau)
    fourier_freqs = fourier_freqs[1:]

    psd = np.ones(shape=(n_times,))
    psd = np.array([F(f, f_k, alpha, delta_nu) for f in fourier_freqs])

    complex_matrix = np.exp(1j*np.random.uniform(0, 2*np.pi, size=psd.shape))
    z = np.sqrt(psd) * complex_matrix
    tod = np.fft.irfft(z) 

    return tod


def H(omega, omega_0, beta):
    exponent = (1-beta) / beta
    return (omega_0 / omega)**exponent

def psd_func(f, omega, f_k, alpha, omega_0, beta, delta_nu):
    psd_comp = F(f, f_k, alpha, delta_nu) * H(omega, omega_0, beta)
    
    return psd_comp

def generate_psd(f_k=1, alpha=2, beta=0.5, receiver_sample_rate=20e6, n_times=2**14, tau=1, n_channels=2**14, normalise=True):
    """
    Generates time ordered data with 1/f fluctuations in time and decorrelation in frequency

    f_k - characteristic frequency of time variations
    alpha - spectral index of time variations
    beta - gain decorrelation parameter
    receiver_sample_rate - sample rate of receiver
    n_times - length of time series
    tau - integration time or temporal resolution
    n_channels - number of frequency channels
    normalise - (bool) best not to normalise at the moment

    returns
    tod - (n_times, n_channels) array with 1/f variations to be used agaisnt time-series data
    """
    delta_nu = receiver_sample_rate / n_channels
    fourier_freqs = np.linspace(1/(n_times*tau), 1/(2*tau), n_times)

    fourier_omegas = np.linspace(1/(n_channels*delta_nu), 1/(delta_nu*2), n_channels)

    omega_0 = 1 / (n_channels * delta_nu)

    #print(omegas)
    #grid_freq, grid_omega = np.meshgrid(fourier_freqs, fourier_omegas)
    grid_omega, grid_freq = np.meshgrid(fourier_omegas, fourier_freqs)

    psd_array = psd_func(grid_freq, grid_omega, f_k, alpha, omega_0, beta, delta_nu)

    if normalise:
        f = lambda omega : omega_0 * np.sinc(np.pi*delta_nu*tau)**2*H(omega, omega_0, beta)
        K = np.sum(np.array([f(omega) for omega in fourier_omegas]))
        psd_array /= K

    complex_matrix = np.exp(1j*np.random.uniform(0, 2*np.pi, size=psd_array.shape))
    #complex_matrix = np.random.uniform(0, 2*np.pi, size=psd.shape) + 1j * np.random.uniform(0, 2*np.pi, size=psd.shape)
    z = np.sqrt(psd_array) * complex_matrix

    
    #tod = np.fft.ifft2(z)
    tod = scipy.fft.ifft2(z, workers=-1)
    tod = np.abs(tod)

    return tod


def one_over_f(f, white_noise_level, f_knee, alpha):
    psd = white_noise_level * ( 1+ ((f_knee / f)**alpha))
    return psd

def generate_one_over_f_time_series(length, delta_t, white_noise_level, f_knee, alpha):
    """
    Generates a time series with white and one over f noise inverse spectrum.

    Arguments:
    length -- length of the timeseries (best if 2**n - 2 for fastest performance)
    delta_t -- spacing between the time series
    white_noise_level -- white noise variance
    knee_frequency -- knee frequency where 1/f equels white noise variance
    alpha -- spectral index of the pink noise

    """
    fourier_freqs = np.fft.rfftfreq(length+2, d=delta_t)
    fourier_freqs = fourier_freqs[1:]
    random_phases = np.random.uniform(0, 2*np.pi, size=fourier_freqs.shape)
    complex_psd = one_over_f(fourier_freqs, white_noise_level, f_knee, alpha) * np.exp(1j*random_phases) 
    varying_signal = np.fft.irfft(complex_psd) * length
    return varying_signal

def total_radiometric_power(source, receiver):
    admitance = np.sqrt(1 - np.abs(receiver.reflection_coefficients)**2) / (1 - receiver.reflection_coefficients*source.reflection_coefficients)

    source_term = source.temperatures * (np.abs(admitance)**2) * (1-np.abs(source.reflection_coefficients)**2)

    uncorelated_term = receiver.t_unc * (np.abs(admitance)**2) * (source.reflection_coefficients**2)

    sin_term = np.abs(admitance)*np.abs(source.reflection_coefficients) * receiver.t_sin * np.sin(np.angle(admitance*source.reflection_coefficients))

    cos_term = np.abs(admitance)*np.abs(source.reflection_coefficients) * receiver.t_cos * np.cos(np.angle(admitance*source.reflection_coefficients))

    total_power = source_term + uncorelated_term + cos_term + sin_term + receiver.t_n
    return np.abs(total_power, dtype='float64')


def add_radiometric_noise(radiometric_power, multiplier):
    delta_p = radiometric_power * multiplier # multiplier = sample_rate / fft_length
    noisy = radiometric_power + np.random.normal(loc=0, scale=delta_p) # 
    return noisy

def add_white_thermal_noise(t_sys, delta_nu, tau):
    delta_t = t_sys / np.sqrt(delta_nu * tau)
    t_sys += np.random.normal(loc=0, scale=delta_t)
    return t_sys

def compute_frame(receiver_gain, cw_amplitude, receiver, cw_source, source):  
    initial_spectrum = total_radiometric_power(source, receiver)
    #initial_spectrum = np.abs(add_radiometric_noise(initial_spectrum, receiver.sample_rate / receiver.n_freq_channels)) #receiver.sample_rate / receiver.n_freq_channels
    initial_spectrum = np.abs(add_white_thermal_noise(initial_spectrum, delta_nu=receiver.channel_width, tau=receiver.delta_t_spectrum))
    complex_time_series = np.fft.ifft(np.sqrt(initial_spectrum))

    if cw_source != None:
        complex_time_series += cw_amplitude*cw_source.phase_noise_sine_signal(receiver.n_freq_channels, 1/receiver.sample_rate)
    
    complex_time_series *= receiver.window_function

    final_spectrum = np.abs(np.fft.fft(complex_time_series))**2 # add normalising constants so it isn't shrank

    return receiver_gain * final_spectrum

class Source:
    def __init__(self, temperatures, reflection_coefficients, frequencies):
        self.temperatures, self.reflection_coefficients = temperatures, reflection_coefficients
        self.frequencies = frequencies

        if isinstance(self.temperatures, float) or isinstance(self.temperatures, int):
            self.temperatures = self.temperatures * np.ones(len(self.frequencies))
        elif isinstance(self.reflection_coefficients, float) or isinstance(self.reflection_coefficients, int) \
              or isinstance(self.reflection_coefficients, complex):
            self.reflection_coefficients = self.reflection_coefficients * np.ones(len(self.frequencies))
        else:
            pass

    def fourier_series_fit(x, y, order):
        pass

    def extrapolate_ref_coeff_to_band(self, band_frequencies, polynomial_order=30, plot_order_residuals=False):
        """
        Converts original input reflection coefficients and frequencies to the observing band and to same length using polyfit

        Need to consider the amplitude and phase

        """

        amps = np.abs(self.reflection_coefficients)
        phases = np.angle(self.reflection_coefficients)

        poly_amps = np.poly1d(np.polyfit(self.frequencies, amps, deg=polynomial_order))
        poly_phase = np.poly1d(np.polyfit(self.frequencies, phases, deg=polynomial_order))

        new_amps = poly_amps(band_frequencies)
        new_phases = poly_phase(band_frequencies)

        if plot_order_residuals:
            fig, axes = plt.subplots(nrows=2)
            axes[0].plot(self.frequencies, amps, label='Input-Amps', c='green')
            axes[2].plot(self.frequencies, phases, label='Input-Phase', c='green', linestyle='--')
            axes[0].plot(self.frequencies, poly_amps(self.frequencies), label='Model-Amps', c='red')
            axes[2].plot(self.frequencies, poly_phase(self.frequencies), label='Model-Phase', c='red', linestyle='--')
            axes[1].plot(self.frequencies, amps - poly_amps(self.frequencies), label='Amp Residual')
            axes[3].plot(self.frequencies, phases - poly_phase(self.frequencies), label='Phase Residual')
            axes[0].legend()
            axes[1].legned()
            axes[2].legend()
            axes[3].legned()
            axes[1].set_xlabel(r'$\nu$ [MHz]')
            axes[0].set_ylabel(r'$|S_{11}|$')
            axes[1].set_ylabel(r'$|S_{11}|$ Residuals')
            axes[2].set_ylabel(r'$\angle S_{11}$')
            axes[3].set_ylabel(r'$\angle S_{11}$ Residuals')
            fig.tight_layout()
            fig.show()

        self.reflection_coefficients = new_amps * np.exp(1j*new_phases)
        self.frequencies = band_frequencies
        return
    
class TerminatedCable:
    def __init__(self, physical_temperature, frequencies, termination='Open', epsilon=1,cable_length=10, mag_s12=1, termination_reflection_coeffs=1):
        self.termination_type = termination
        c = 299792458
        self.frequencies = frequencies
        self.temperatures = physical_temperature * np.ones(len(self.frequencies))
        if termination=='Open':
            self.reflection_coefficients = (mag_s12**2) * np.exp(-2j*self.frequencies*cable_length*epsilon/c)
        elif termination=='Shorted':
            self.reflection_coefficients = -(mag_s12**2) * np.exp(-2j*self.frequencies*cable_length*epsilon/c)
        else:
            self.reflection_coefficients = (mag_s12**2) * np.exp(-2j*self.frequencies*cable_length*epsilon/c)* np.exp(-1j*np.pi) * termination_reflection_coeffs

        pass
    

class TerminatedCableOld:
    def __init__(self, termination='load', cable_length=30, physical_temperature=300, cable_s_params=[], termination_impedance=50, reference_impedance=50,
                 cable_temperature=300):
        self.termination = termination
        s11, s12, s21, s22 = cable_s_params
        if termination == "load":
            src_reflection_coeff = () / ()

        self.reflection_coefficient = s11 + ((s12 * s21 * src_reflection_coeff) / (1 - (s22 * src_reflection_coeff)))

        cable_gain = 0

        pass


class VectorNetworkAnalyser:
    def __init__(self, start, stop):
        pass

class LogPowerMeter:
    def __init__(self, characteristic_frequency, sample_rate, alpha, bit_depth, physical_temperature, bandwidth,
                 slope, max_voltage=3.3):
        self.characteristic_frequency = characteristic_frequency
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.bit_depth = bit_depth
        self.physical_temperature = physical_temperature
        self.bandwidth = bandwidth
        self.measurement_levels = 2**self.bit_depth
        self.slope = slope
        self.max_voltage=max_voltage
        pass

    def measure_cw_power(self, cw_v_amplitude):
        cw_pwr = cw_v_amplitude**2
        return cw_pwr


class BasicPowerMeter:
    def __init__(self, linear_scale_factor, power_offset,
                 characteristic_frequency, alpha, sample_rate,
                 white_noise_level):
        self.linear_scale_factor = linear_scale_factor
        self.power_offset = power_offset
        self.characteristic_frequency = characteristic_frequency
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.white_noise_level = white_noise_level
        pass
    def measure_cw_power(self, cw_v_amplitude):
        cw_pwr = cw_v_amplitude**2
        return cw_pwr*self.power_offset + self.power_offset
    
    def add_white_noise(self, powers):
        noise = np.random.normal(loc=0, scale=self.white_noise_level, size=powers.shape)
        return powers + noise
    

class CW_Source:
    def __init__(self, initial_cw_amplitude, oscilator_frequency, characteristic_frequency, alpha,
                 phase_white_noise, phase_knee_frequency, phase_alpha):
        self.initial_cw_amplitude = initial_cw_amplitude
        self.characteristic_frequency = characteristic_frequency
        self.alpha = alpha
        self.phase_white_noise = phase_white_noise
        self.phase_knee_frequency = phase_knee_frequency
        self.phase_alpha = phase_alpha
        self.oscilator_frequency = oscilator_frequency
        pass

    def set_baseband_frequency(self, receiver):
        #self.base_band_frequency = receiver.centre_frequency - self.oscilator_frequency
        self.base_band_frequency = receiver.centre_frequency - self.oscilator_frequency

    def phase_noise_sine_signal(self, length, delta_t):
        phase_noise = generate_one_over_f_time_series(length, delta_t, self.phase_white_noise,
                                                 self.phase_knee_frequency, self.phase_alpha)
        times = np.arange(length) * delta_t

        phases = (2*np.pi*self.base_band_frequency*times) + phase_noise

        return np.exp(1j*(phases))


class SDR_Receiver:
    def __init__(self, characteristic_frequency, alpha, centre_frequency, n_freq_channels, sample_rate,
                 t_unc, t_sin, t_cos, t_n, reflection_coefficients, window_function, beta=0):
        self.characteristic_frequency, self.alpha = characteristic_frequency, alpha
        self.centre_frequency, self.n_freq_channels = centre_frequency, n_freq_channels
        self.sample_rate = sample_rate
        self.reflection_coefficients = reflection_coefficients
        self.window_function = window_function
        self.delta_t_spectrum = self.n_freq_channels / self.sample_rate
        self.channel_width = self.sample_rate / self.n_freq_channels
        self.frequencies = np.linspace(self.centre_frequency - (self.sample_rate/2), self.centre_frequency + (self.sample_rate/2), self.n_freq_channels)
        self.beta = beta

        if isinstance(t_unc, int) or isinstance(t_unc, float):
            self.t_unc = t_unc * np.ones(shape=n_freq_channels)
        if isinstance(t_cos, int) or isinstance(t_cos, float):
            self.t_cos = t_cos * np.ones(shape=n_freq_channels)
        if isinstance(t_sin, int) or isinstance(t_sin, float):
            self.t_sin = t_sin * np.ones(shape=n_freq_channels)
        if isinstance(t_n, int) or isinstance(t_n, float):
            self.t_n = t_n * np.ones(shape=n_freq_channels)
        if isinstance(self.reflection_coefficients, float) or isinstance(self.reflection_coefficients, int) \
              or isinstance(self.reflection_coefficients, complex):
            self.reflection_coefficients = self.reflection_coefficients * np.ones(shape=n_freq_channels)

        if self.window_function == 'Blackman':
            self.window_function = np.blackman(self.n_freq_channels)
        elif self.window_function == 'Bartlett':
            self.window_function = np.bartlett(self.n_freq_channels)
        elif self.window_function == 'Hamming':
            self.window_function = np.hamming(self.n_freq_channels)
        elif self.window_function == 'Hanning':
            self.window_function = np.hanning(self.n_freq_channels)
        else:
            self.window_function = np.ones(self.n_freq_channels)

        pass



def generate_switch_array(n_integrations, switch_cycle_period, integration_time, total_time,switch_obs_fraction=[]):
    max_index = int(np.ceil(n_integrations))
    n_c = int(np.ceil(switch_cycle_period / integration_time))
    n_i = [int(n_c * n) for n in switch_obs_fraction]
    rmd = n_c - sum(n_i)
    n_i[0] += rmd

    base_cycle = [i * np.ones(shape=(n,)) for i, n in enumerate(n_i)]
    base_cycle = np.concatenate(base_cycle)
    switch_array = np.tile(base_cycle, reps=int(np.ceil(total_time / switch_cycle_period)))
    switch_array = switch_array[:max_index]
    return switch_array.astype(dtype=int)


class TimeStreamGenerator:
    def __init__(self, integration_time=5, simulation_time = 60, bandwidth=20e6, centre_frequency=70e6, n_freq_channels=2**12):
        self.n_freq_channels = n_freq_channels
        self.centre_frequency = centre_frequency
        self.bandwidth = bandwidth
        self.integration_time = integration_time
        self.simulation_time = simulation_time
        
        self.n_frames = int(simulation_time * self.bandwidth / self.n_freq_channels)
        self.n_integrations = simulation_time / integration_time
        self.frames_per_integration = np.floor(self.n_frames / self.n_integrations)
        self.delta_t = self.simulation_time / self.n_frames

        pass

    def set_nframes(self, n_frames):
        self.n_frames = n_frames
        
        self.simulation_time = self.n_frames * self.n_freq_channels / self.bandwidth

        self.n_integrations = self.simulation_time / self.integration_time
        self.frames_per_integration = np.floor(self.n_frames / self.n_integrations)
        self.delta_t = self.simulation_time / self.n_frames

        print('----====----')
        print(f'simulation time is now - {self.simulation_time} seconds')
        print('----++++----')

        pass


    def generate_simulated_data(self, obs_source, cw_source, receiver, save_data=False, savepath='', title='',switching=False, 
                                          switch_sources=[], switch_obs_fraction=[1/3,1/3,1/3], switch_cycle_period=60,
                                          power_meter=None, plot_spectra=True, save_into_object = True, return_list=False):
        """
        Runs the simulation and returns measured values into the initiated object if save_into_object=True.

        If beta != 0 then the gains are 2D in time and frequency. This can be memory restricted
        as each row corresponds to a single spectra frame in the receiver. For less memory
        restricted version use generate_simulated_data_restricted_gains.


        """
        print('   generating data')
        if receiver.beta == 0:
            system_gains = 1 + generate_F_psd(f_k=receiver.characteristic_frequency, alpha=receiver.alpha, receiver_sample_rate=receiver.sample_rate,
                                              n_times=self.n_frames, tau=self.delta_t, n_channels=receiver.n_freq_channels) # n_frames

            system_gains = np.abs(system_gains)
        else:
            system_gains = 1 + generate_psd(receiver.characteristic_frequency, alpha=receiver.alpha, beta=receiver.beta,
                                            receiver_sample_rate=receiver.sample_rate, n_times=self.n_frames, tau=self.delta_t,
                                            n_channels=receiver.n_freq_channels)
        
        if cw_source is None:
            cw_amplitudes = np.zeros(shape=system_gains.shape)
        else:
            cw_source.set_baseband_frequency(receiver)
            cw_amplitudes = cw_source.initial_cw_amplitude * (1 + generate_F_psd(f_k = cw_source.characteristic_frequency, alpha=cw_source.alpha,
                                                                                 receiver_sample_rate=receiver.sample_rate, n_times = self.n_frames,
                                                                                 tau=self.delta_t, n_channels=receiver.n_freq_channels))
            cw_amplitudes = np.abs(cw_amplitudes)

        integrated_spectra = np.ones(shape=(int(np.ceil(self.n_integrations)), receiver.n_freq_channels))

        if switching:
            switches_list = generate_switch_array(n_integrations=self.n_integrations,
                                                  switch_cycle_period=switch_cycle_period,
                                                  total_time=self.simulation_time,
                                                  switch_obs_fraction=switch_obs_fraction,
                                                  integration_time=self.integration_time)
        
        for i in range(np.ceil(int(self.n_integrations))+1): # mind the ceiling and the fact this isnt an integer. See how it works with indices
            #print(i)
            if switching:
                switch_index = switches_list[i]
                obs_source = switch_sources[switch_index]

            if i == int(np.floor(self.n_integrations)):
                g_list = system_gains[int(i*self.frames_per_integration): int(self.n_frames-1)]
                #print('final g_list shape - ', g_list.shape)
                #print('final i = ', i)
                cw_amp_list = cw_amplitudes[int(i*self.frames_per_integration): int(self.n_frames-1)]
            else:
                g_list = system_gains[int(i*self.frames_per_integration): int(i*self.frames_per_integration + self.frames_per_integration)]
                cw_amp_list = cw_amplitudes[int(i*self.frames_per_integration): int(i*self.frames_per_integration + self.frames_per_integration)]
            
            with multiprocessing.Pool() as pool:
                frames = pool.starmap(compute_frame, zip(g_list, cw_amp_list, itertools.repeat(receiver),
                                                         itertools.repeat(cw_source), itertools.repeat(obs_source)))
                frames = np.array(frames)
                int_spectra = np.mean(frames, axis=0)
                try:
                    integrated_spectra[i] = int_spectra
                except:
                    #print('additional i = ', i)
                    pass
            pass
        

        spectra_times = np.linspace(0, self.simulation_time, int(np.ceil(self.n_integrations)))
        obs_freqs_mhz = receiver.frequencies / 1e6
        extent = [obs_freqs_mhz[0] ,obs_freqs_mhz[-1] , spectra_times[-1], spectra_times[0]]
        frame_times = np.linspace(0, self.simulation_time, self.n_frames)

        if power_meter is not None:
            pm_cw_measurements = np.ones(shape=len(integrated_spectra[:,0]))
            pm_gains = 1 + generate_F_psd(f_k=power_meter.characteristic_frequency,
                                            alpha=power_meter.alpha,
                                            receiver_sample_rate=receiver.sample_rate,
                                            n_channels=receiver.n_freq_channels,
                                            tau=1 / receiver.sample_rate,
                                            n_times=self.n_frames
                                            )
            base_measured_powers = power_meter.measure_cw_power(cw_amplitudes)
            base_measured_powers = power_meter.add_white_noise(base_measured_powers)
            base_measured_powers *= pm_gains

            split_cw_amplitudes = np.array_split(base_measured_powers, len(pm_cw_measurements), )
            pm_cw_measurements = np.array([np.mean(s) for s in split_cw_amplitudes])

        if save_data:
            with h5py.File(savepath+title, mode='w') as file:
                simulation_params = file.create_group('simulation_params')
                data = file.create_group('data')

                simulation_params.create_dataset('Receiver_Gains', data=system_gains, dtype=system_gains.dtype)
                simulation_params.create_dataset('CW_Amplitudes', data=cw_amplitudes, dtype=cw_amplitudes.dtype)
                simulation_params.create_dataset('Frame_Times', data=frame_times, dtype=frame_times.dtype)

                data.create_dataset('Spectra', data=integrated_spectra, dtype=integrated_spectra.dtype)
                data.create_dataset('Times', data=spectra_times, dtype=spectra_times.dtype)
                data.create_dataset('Frequencies_MHz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)

                if switching:
                    data.create_dataset('Switch List', data=switches_list, dtype=switches_list.dtype)
                
                if power_meter is not None:
                    data.create_dataset('PM_Measurements', data=pm_cw_measurements, dtype=pm_cw_measurements.dtype)
                    simulation_params.create_dataset('PM_Gains', data=pm_gains, dtype=pm_gains.dtype)

        if save_into_object:
            self.receiver_gains = system_gains
            self.cw_amplitudes = cw_amplitudes
            self.integrated_spectra = integrated_spectra
            self.times = spectra_times
            self.frame_times = frame_times
            self.frequencies_mhz = obs_freqs_mhz
            if switching:
                self.switch_list = switches_list
            if power_meter is not None:
                self.pm_measurements = pm_cw_measurements
                self.pm_gains = pm_gains

        if plot_spectra:
            plt.imshow(10*np.log10(integrated_spectra), aspect='auto', extent=extent)
            plt.ylabel('Time [s]')
            plt.xlabel('Frequency [MHz]')
            plt.tight_layout()
            plt.show()
        #with h5py.File(name='Test_Export/Test.hd5f', mode='a') as file:
        #    file.create_dataset('spectra', data=integrated_spectra, dtype=integrated_spectra.dtype)
        #    file.create_dataset('times', data=times, dtype=times.dtype)
        #    file.create_dataset('obs_freqs_mhz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)

        print('----====----')
        print(system_gains.shape, ' - system gains')
        print(cw_amplitudes.shape, ' - cw amplitudes')
        print(self.n_integrations, ' - n_integrations')
        print(self.n_frames, ' - nframes')
        print(integrated_spectra.shape, ' - integrated spectra')

        if return_list:
            rt_list = []
            rt_list.append(system_gains)
            rt_list.append(cw_amplitudes)
            rt_list.append(integrated_spectra)
            rt_list.append(spectra_times)
            rt_list.append(frame_times)
            rt_list.append(obs_freqs_mhz)
            if switching and (power_meter is not None):
                rt_list.append(switches_list)
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'switch_list':6,
                     'pm_measurements':7,
                     'pm_gains':8,
                     'dictionary':9}
                rt_list.append(d)
                return rt_list
            
            elif switching:
                rt_list.append(switches_list)
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'switch_list':6,
                     'dictionary':7}
                rt_list.append(d)
                return rt_list
            
            elif power_meter is not None:
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'pm_measurements':6,
                     'pm_gains':7,
                     'dictionary':8}
                rt_list.append(d)
                return rt_list
            
            else:
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'dictionary':6}
                rt_list.append(d)
                return rt_list


        pass
    
    def generate_simulated_data_restricted_gains(self, obs_source, cw_source, receiver, save_data=False, savepath='', title='',switching=False, 
                                          switch_sources=[], switch_obs_fraction=[1/3,1/3,1/3], switch_cycle_period=60,
                                          power_meter=None, plot_spectra=True, save_into_object = True, return_list=False):
        """
        Operates in the same way as generate_simulated_data but with the gains applied to each integration so that
        longer simulated times can be ran.
        """
        print('   generating data')
        n_ints = int(np.ceil(self.n_integrations))
        dt_int = self.integration_time
        if receiver.beta == 0:
            system_gains = 1 + generate_F_psd(f_k=receiver.characteristic_frequency, alpha=receiver.alpha, receiver_sample_rate=receiver.sample_rate,
                                              n_times=n_ints, tau=dt_int, n_channels=receiver.n_freq_channels) # n_frames

            system_gains = np.abs(system_gains)
        else:
            system_gains = 1 + generate_psd(receiver.characteristic_frequency, alpha=receiver.alpha, beta=receiver.beta,
                                            receiver_sample_rate=receiver.sample_rate, n_times=n_ints, tau=dt_int,
                                            n_channels=receiver.n_freq_channels)
        
        if cw_source is None:
            cw_amplitudes = np.zeros(shape=system_gains.shape)
        else:
            cw_source.set_baseband_frequency(receiver)
            cw_amplitudes = cw_source.initial_cw_amplitude * (1 + generate_F_psd(f_k = cw_source.characteristic_frequency, alpha=cw_source.alpha,
                                                                                 receiver_sample_rate=receiver.sample_rate, n_times = self.n_frames,
                                                                                 tau=self.delta_t, n_channels=receiver.n_freq_channels))
            cw_amplitudes = np.abs(cw_amplitudes)

        integrated_spectra = np.ones(shape=(int(np.ceil(self.n_integrations)), receiver.n_freq_channels))

        if switching:
            switches_list = generate_switch_array(n_integrations=self.n_integrations,
                                                  switch_cycle_period=switch_cycle_period,
                                                  total_time=self.simulation_time,
                                                  switch_obs_fraction=switch_obs_fraction,
                                                  integration_time=self.integration_time)
        
        for i in range(np.ceil(int(self.n_integrations))+1): # mind the ceiling and the fact this isnt an integer. See how it works with indices
            print(i)
            if switching:
                switch_index = switches_list[i]
                obs_source = switch_sources[switch_index]

            if i == int(np.floor(self.n_integrations)):
                
                #g_list = system_gains[int(i*self.frames_per_integration): int(self.n_frames-1)]
                #print('final g_list shape - ', g_list.shape)
                #print('final i = ', i)
                cw_amp_list = cw_amplitudes[int(i*self.frames_per_integration): int(self.n_frames-1)]
                g_list = [1] * len(cw_amp_list)
            else:
                #g_list = system_gains[int(i*self.frames_per_integration): int(i*self.frames_per_integration + self.frames_per_integration)]
                cw_amp_list = cw_amplitudes[int(i*self.frames_per_integration): int(i*self.frames_per_integration + self.frames_per_integration)]
                g_list = [1] * len(cw_amp_list)
            
            with multiprocessing.Pool() as pool:
                frames = pool.starmap(compute_frame, zip(g_list, cw_amp_list, itertools.repeat(receiver),
                                                         itertools.repeat(cw_source), itertools.repeat(obs_source)))
                frames = np.array(frames)
                int_spectra = np.mean(frames, axis=0)
                int_spectra *= system_gains[i]
                try:
                    integrated_spectra[i] = int_spectra
                except:
                    #print('additional i = ', i)
                    pass
            pass
        

        spectra_times = np.linspace(0, self.simulation_time, int(np.ceil(self.n_integrations)))
        obs_freqs_mhz = receiver.frequencies / 1e6
        extent = [obs_freqs_mhz[0] ,obs_freqs_mhz[-1] , spectra_times[-1], spectra_times[0]]
        frame_times = np.linspace(0, self.simulation_time, self.n_frames)

        if power_meter is not None:
            pm_cw_measurements = np.ones(shape=len(integrated_spectra[:,0]))
            pm_gains = 1 + generate_F_psd(f_k=power_meter.characteristic_frequency,
                                            alpha=power_meter.alpha,
                                            receiver_sample_rate=receiver.sample_rate,
                                            n_channels=receiver.n_freq_channels,
                                            tau=1 / receiver.sample_rate,
                                            n_times=self.n_frames
                                            )
            base_measured_powers = power_meter.measure_cw_power(cw_amplitudes)
            base_measured_powers = power_meter.add_white_noise(base_measured_powers)
            base_measured_powers *= pm_gains

            split_cw_amplitudes = np.array_split(base_measured_powers, len(pm_cw_measurements), )
            pm_cw_measurements = np.array([np.mean(s) for s in split_cw_amplitudes])

        if save_data:
            with h5py.File(savepath+title, mode='w') as file:
                simulation_params = file.create_group('simulation_params')
                data = file.create_group('data')

                simulation_params.create_dataset('Receiver_Gains', data=system_gains, dtype=system_gains.dtype)
                simulation_params.create_dataset('CW_Amplitudes', data=cw_amplitudes, dtype=cw_amplitudes.dtype)
                simulation_params.create_dataset('Frame_Times', data=frame_times, dtype=frame_times.dtype)

                data.create_dataset('Spectra', data=integrated_spectra, dtype=integrated_spectra.dtype)
                data.create_dataset('Times', data=spectra_times, dtype=spectra_times.dtype)
                data.create_dataset('Frequencies_MHz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)

                if switching:
                    data.create_dataset('Switch List', data=switches_list, dtype=switches_list.dtype)
                
                if power_meter is not None:
                    data.create_dataset('PM_Measurements', data=pm_cw_measurements, dtype=pm_cw_measurements.dtype)
                    simulation_params.create_dataset('PM_Gains', data=pm_gains, dtype=pm_gains.dtype)

        if save_into_object:
            self.receiver_gains = system_gains
            self.cw_amplitudes = cw_amplitudes
            self.integrated_spectra = integrated_spectra
            self.times = spectra_times
            self.frame_times = frame_times
            self.frequencies_mhz = obs_freqs_mhz
            if switching:
                self.switch_list = switches_list
            if power_meter is not None:
                self.pm_measurements = pm_cw_measurements
                self.pm_gains = pm_gains

        

        if plot_spectra:
            plt.imshow(10*np.log10(integrated_spectra), aspect='auto', extent=extent)
            plt.ylabel('Time [s]')
            plt.xlabel('Frequency [MHz]')
            plt.tight_layout()
            plt.show()
        #with h5py.File(name='Test_Export/Test.hd5f', mode='a') as file:
        #    file.create_dataset('spectra', data=integrated_spectra, dtype=integrated_spectra.dtype)
        #    file.create_dataset('times', data=times, dtype=times.dtype)
        #    file.create_dataset('obs_freqs_mhz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)

        print('----====----')
        print(system_gains.shape, ' - system gains')
        print(cw_amplitudes.shape, ' - cw amplitudes')
        print(self.n_integrations, ' - n_integrations')
        print(self.n_frames, ' - nframes')
        print(integrated_spectra.shape, ' - integrated spectra')
        
        if return_list:
            rt_list = []
            rt_list.append(system_gains)
            rt_list.append(cw_amplitudes)
            rt_list.append(integrated_spectra)
            rt_list.append(spectra_times)
            rt_list.append(frame_times)
            rt_list.append(obs_freqs_mhz)
            if switching and (power_meter is not None):
                rt_list.append(switches_list)
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'switch_list':6,
                     'pm_measurements':7,
                     'pm_gains':8,
                     'dictionary':9}
                rt_list.append(d)
                return rt_list
            
            elif switching:
                rt_list.append(switches_list)
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'switch_list':6,
                     'dictionary':7}
                rt_list.append(d)
                return rt_list
            
            elif power_meter is not None:
                rt_list.append(pm_cw_measurements)
                rt_list.append(pm_gains)
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'pm_measurements':6,
                     'pm_gains':7,
                     'dictionary':8}
                rt_list.append(d)
                return rt_list
            
            else:
                d = {'receiver_gains':0,
                     'cw_amplitudes':1,
                     'integrated_spectra':2,
                     'times':3,
                     'frame_times':4,
                     'frequencies_mhz':5,
                     'dictionary':6}
                rt_list.append(d)
                return rt_list

        pass

    def read_in_from_h5py(self, filepath):
        with h5py.File(filepath, mode='r') as file:
            simulation_params = file['simulation_params']
            data = file['data']

            self.receiver_gains = simulation_params['Receiver_Gains'][()]
            self.cw_amplitudes = simulation_params['CW_Amplitudes'][()]
            self.frame_times = simulation_params['Frame_Times'][()]

            self.integrated_spectra = data['Spectra'][()]
            self.times = data['Times'][()]
            self.frequencies_mhz = data['Frequencies_MHz'][()]

            try:
                self.switch_list = data['Switch List']
            except:
                print('No Switching')
            try:
                self.pm_measurements = data['PM_Measurements'][()]
                self.pm_gains = simulation_params['PM_Gains'][()]
            except:
                print('No Powermeter Measurement')
        pass



if __name__ == "__main__":
    receiver = SDR_Receiver(characteristic_frequency=50,  alpha=2, centre_frequency=70e6, n_freq_channels=2**12, sample_rate=20e6,
                            t_unc=20, t_sin=20, t_cos=100, t_n=200, reflection_coefficients=0.5, window_function='Blackman', beta=0.0)
    
    target = TerminatedCable(termination='Open', cable_length=30, physical_temperature=300, frequencies=receiver.frequencies, epsilon=1.5)

    spectra = total_radiometric_power(target, receiver)

    #cw_source = CW_Source(initial_cw_amplitude=5e-2, oscilator_frequency=62e6,
    #                      characteristic_frequency=200, alpha=1, phase_white_noise=1e-9, phase_knee_frequency=2e6,
    #                      phase_alpha=1)

    #generator = TimeStreamGenerator(integration_time=0.1, simulation_time=60.0, bandwidth=receiver.sample_rate, centre_frequency=receiver.centre_frequency, n_freq_channels=receiver.n_freq_channels)

    #load_1 = Source(200, 0,frequencies=receiver.frequencies)
    #load_2 = Source(2000, 0, frequencies=receiver.frequencies)

    #power_meter = BasicPowerMeter(linear_scale_factor=0.1, power_offset=2, characteristic_frequency=1e0, alpha=2, sample_rate=5,
    #                              white_noise_level=1e-6)

    #generator.set_nframes(2**14)

    #spectra = generator.generate_simulated_data(target, cw_source, receiver, switching=True, switch_sources=[load_1, load_2, target],
    #                                            switch_obs_fraction=[1/3, 1/3, 1/3], switch_cycle_period=10, power_meter=power_meter)
    
    pass