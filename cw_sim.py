import numpy as np
import matplotlib.pyplot as plt
import datetime

def F(f, pink_variance, alpha):
    return pink_variance*((1/f)**alpha)

def wn_f_PSD(f, white_noise_variance, pink_variance, alpha):
    return white_noise_variance+pink_variance*((1 / f)**alpha)

def one_over_f_timestream(pink_variance, alpha, sample_rate, number_of_samples):
    frequncies = np.fft.rfftfreq(number_of_samples+2, d=1/sample_rate)[1:]
    random_phases = np.random.uniform(0, 2*np.pi, size=len(frequncies))
    complex_psd = F(frequncies, pink_variance, alpha) * [np.cos(random_phases) + np.sin(random_phases)*1j] 
    varying_signal = np.fft.irfft(complex_psd)[0]
    return varying_signal

def white_one_over_f_timestream(white_variance, pink_variance, alpha, sample_rate, number_of_samples):
    frequncies = np.fft.rfftfreq(number_of_samples+2, d=1/sample_rate)[1:]
    random_phases = np.random.uniform(0, 2*np.pi, size=len(frequncies))
    complex_psd = wn_f_PSD(frequncies, 
                           white_variance, 
                           pink_variance, alpha) * [np.cos(random_phases) + np.sin(random_phases)*1j] 
    varying_signal = np.fft.irfft(complex_psd)[0]
    return varying_signal

def cw_isolation_channel_difference(spectra, integrating_side_channels, number_of_sys_temp_channels):
    cw_index = np.argmax(spectra)
    brightness_across_channels = np.sum(spectra[cw_index-integrating_side_channels:cw_index+integrating_side_channels])
    lower_system_temp_average = np.mean(spectra[cw_index-integrating_side_channels-1-number_of_sys_temp_channels:cw_index-integrating_side_channels-1])
    higher_system_temp_average = np.mean(spectra[cw_index+integrating_side_channels+1:cw_index+integrating_side_channels+1+number_of_sys_temp_channels])
    avg_sys_power_per_channel = np.mean([lower_system_temp_average, higher_system_temp_average])
    
    isolated_cw_power = brightness_across_channels - (1+(2*integrating_side_channels))*avg_sys_power_per_channel
    return isolated_cw_power

def cw_isolation_naiive_linear_fit(spectra, integrating_side_channels, number_of_sys_temp_channels):
    cw_index = np.argmax(spectra)
    brightness_across_channels = np.sum(spectra[cw_index-integrating_side_channels:cw_index+integrating_side_channels])
    lower_system_temp_average = np.mean(spectra[cw_index-integrating_side_channels-1-number_of_sys_temp_channels:cw_index-integrating_side_channels-1])
    lower_index_value = cw_index - integrating_side_channels -1 - (number_of_sys_temp_channels/2)
    higher_system_temp_average = np.mean(spectra[cw_index-integrating_side_channels-1-number_of_sys_temp_channels:cw_index-integrating_side_channels-1])
    higher_index_value = cw_index + integrating_side_channels + 1 + (number_of_sys_temp_channels /2)
    gradient = (higher_system_temp_average - lower_system_temp_average) / (higher_index_value - lower_index_value)
    intercept = higher_system_temp_average - (gradient * higher_index_value)

    for i in np.arange(cw_index-integrating_side_channels, cw_index + integrating_side_channels+1):
        background = i*gradient + intercept
        brightness_across_channels -= background
    isolated_cw_power = brightness_across_channels
    return isolated_cw_power


def generate_multiple_qs(time_per_q, number_of_qs, cold_load, hot_load, to_meausure, receiver, cw_source, power_meter, fractional_measurements, file_location):
    sim_params = SimulationParameters(1, fractional_measurements, time_per_q, receiver)
    start_time = datetime.datetime.now()
    simulation_interval = datetime.timedelta(seconds=sim_params.reciever_spectrum_integration_time)
    integrated_spectra, power_meter_measurements, datetimes, switch_time_array, frequencies_channels = TimeOrderedDataGenerator(cold_load, hot_load, to_meausure,
                                                                                                                                                       receiver, cw_source, power_meter, sim_params, start_time).export_time_ordered_data(1/power_meter.sample_rate, 
                                                                                                                                                                                                                              sim_params, file_location, save_to_drive=False)
    print('q1 done')
    for i in np.arange(2,number_of_qs+1):
        int_spec, pmm, dt, sta, fc = TimeOrderedDataGenerator(cold_load, hot_load, to_meausure, receiver, cw_source, power_meter, 
                                                              sim_params, datetimes[-1]+simulation_interval).export_time_ordered_data(1/power_meter.sample_rate, sim_params, file_location, save_to_drive=False)
        integrated_spectra = np.concatenate((integrated_spectra, int_spec))
        power_meter_measurements = np.concatenate((power_meter_measurements, pmm))
        datetimes = np.concatenate((datetimes, dt))
        switch_time_array = np.concatenate((switch_time_array, sta))
        print('q'+str(i)+' done')
    
    np.save(file_location+'Spectra', integrated_spectra)
    np.save(file_location+'PowerMeterMeasurements', power_meter_measurements)
    np.save(file_location+'DateTimes', datetimes)
    np.save(file_location+'SwitchTimes', switch_time_array)
    np.save(file_location+'ChannelFrequencies', frequencies_channels)
    print('Ding Ding: All qs processed and saved under: '+file_location)

    


class Observable:
    def __init__(self, physical_temperature):
        self.physical_temperature = physical_temperature

class OpenCable(Observable):
    def __init__(self, physical_temperature, two_way_delay, two_way_cable_loss_db):
        super().__init__(physical_temperature)
        self.two_way_delay = two_way_delay
        self.two_way_cable_loss_db = two_way_cable_loss_db
        pass

    def compute_reflection_coefficients(self, reciever):
        self.frequencies = np.linspace(-reciever.sample_rate /2 + reciever.centre_frequency, 
                                  reciever.sample_rate /2 + reciever.centre_frequency,
                                  reciever.fft_length)
        
        self.loss_factor = 10**(-self.two_way_cable_loss_db / 10)
        self.frequencies = np.linspace(-reciever.sample_rate/2 + reciever.centre_frequency, 
                                       -reciever.sample_rate/2 + reciever.centre_frequency, reciever.fft_length)

        self.reflection_coefficients = self.loss_factor * np.exp(-1j*2*np.pi*self.frequencies*self.two_way_delay)
        return self.reflection_coefficients
    
    def brightness_temperature(self, reciever):
        return self.physical_temperature * np.ones(reciever.fft_length)


class Load(Observable):
    def __init__(self, physical_temperature):
        super().__init__(physical_temperature)
        self.reflection_coef = 0
        pass
    def compute_reflection_coefficients(self, reciever):
        self.reflection_coefficients = np.zeros(reciever.fft_length)
        return self.reflection_coefficients
    def brightness_temperature(self, reciever):
        return self.physical_temperature * np.ones(reciever.fft_length)

class PowerMeter:
    def __init__(self, feedback_factor_x, pink_noise_variance, sample_rate, alpha, bit_depth, physical_temperature, bandwidth):
        self.feedback_factor_x  = feedback_factor_x
        self.pink_noise_variance = pink_noise_variance
        self.sample_rate = sample_rate
        self.alpha = alpha
        self.bit_depth = bit_depth
        self.physical_temperature = physical_temperature
        self.bandwidth = bandwidth
        self.measurement_levels = 2**self.bit_depth
        pass

    def output_voltage_bits(self, cw_power_watts, gain_fluctuation):
        self.p_in_mw = (cw_power_watts + (self.physical_temperature*self.bandwidth * 1.380649 * 10**-23)) * 10**-3 #mW
        self.p_in_mw = np.random.normal(loc=self.p_in_mw, scale=self.physical_temperature*self.bandwidth*1.380649*10**-26) * gain_fluctuation
        self.p_in_dbm = 10*np.log10(self.p_in_mw)
        self.voltage = -22 * self.feedback_factor_x * (10 ** -3) *(self.p_in_dbm - 15) 
        self.bit_voltage = int(self.measurement_levels * self.voltage / 3.3)
        return self.bit_voltage
    
    def convert_bit_voltgage_to_power(self, power_meter_bits):
        self.voltage = power_meter_bits * 3.3 / self.measurement_levels
        self.p_dbm = (self.voltage / (-22 * self.feedback_factor_x * 10**-3)) + 15
        self.p_mw = 10**(self.p_dbm/10)
        return self.p_mw

class CWSource:
    def __init__(self, pink_noise_variance, alpha, cw_power_dbm, cw_frequency, attenuation_before_receiver_db, receiver):
        self.pink_noise_variance = pink_noise_variance
        self.alpha = alpha
        self.cw_power_dbm = cw_power_dbm
        self.cw_power_watts = 10**(cw_power_dbm / 10) * 10 ** -3
        self.cw_temperature = self.cw_power_watts / (receiver.channel_width * 1.380649*10**-23)
        self.cw_frequency = cw_frequency
        self.attenuation_before_receiver = 10**(attenuation_before_receiver_db/10)
        pass

    def brightness_temperature(self, reciever):
        self.relative_cw_position = (self.cw_frequency - reciever.centre_frequency) / reciever.sample_rate
        if reciever.window == 'Blackman':
            self.fft_filter = np.blackman(reciever.fft_length)
        elif reciever.window == 'Hamming':
            self.fft_filter = np.hamming(reciever.fft_length)
        elif reciever.window == 'Bartlett':
            self.fft_filter = np.bartlett(reciever.fft_length)
        elif reciever.window == 'Hanning':
            self.fft_filter = np.hanning(reciever.fft_length)
        else:
            self.fft_filter = np.ones(reciever.fft_length)
        self.number_of_samples = reciever.sample_rate
        self.times = np.linspace(0, self.number_of_samples / reciever.sample_rate, self.number_of_samples)
        self.cw_freq = reciever.sample_rate * self.relative_cw_position
        self.complex_signal = np.cos(2*np.pi*self.cw_freq*self.times) + np.sin(2*np.pi*self.cw_freq*self.times)*1j

        self.waterfall = []
        self.number_of_ffts = int(np.floor(self.number_of_samples / reciever.fft_length))
        for self.spectra in np.arange(0, self.number_of_ffts):
            self.line = self.complex_signal[self.spectra*reciever.fft_length:(reciever.fft_length*(1+self.spectra))]  * self.fft_filter
            self.spectrum = np.abs(np.fft.fftshift(np.fft.fft(self.line)))**2
            self.waterfall.append(np.abs(np.fft.fftshift(np.fft.fft(self.line)))**2)
        self.summed_spectra = np.sum(self.waterfall, axis=0)
        self.normalised_spectra = self.summed_spectra / np.sum(self.summed_spectra)
        return self.normalised_spectra * self.cw_temperature * self.attenuation_before_receiver


class Reciever:
    def __init__(self, sample_rate, centre_frequency, fft_length, noise_temperature, 
                 uncorrelated_reflected_noise_temp, sin_noise_temp, cos_noise_temp,
                 pink_noise_variance, alpha, window, reflection_coefficients):
        self.sample_rate = sample_rate
        self.centre_frequency = centre_frequency
        self.fft_length = fft_length
        self.noise_temperature = noise_temperature
        self.uncorrelated_reflected_noise_temp = uncorrelated_reflected_noise_temp
        self.sin_noise_temp = sin_noise_temp
        self.cos_noise_temp = cos_noise_temp
        self.nw_params = [uncorrelated_reflected_noise_temp, sin_noise_temp, cos_noise_temp]
        self.pink_noise_variance = pink_noise_variance
        self.alpha = alpha
        self.window = window
        self.channel_width = self.sample_rate / self.fft_length

        if type(reflection_coefficients) == int or type(reflection_coefficients) == float:
            self.reflection_coefficients = reflection_coefficients * np.ones(self.fft_length)
        else:
            self.reflection_coefficients = reflection_coefficients
        pass

    def reciever_nw_spectrum(self):
        self.noise_spectrum = self.noise_temperature * np.ones(self.fft_length)
        self.uncorrelated_reflected_noise_spectrum = self.uncorrelated_reflected_noise_temp * np.ones(self.fft_length)
        self.sin_noise_spectrum = self.sin_noise_temp * np.ones(self.fft_length)
        self.cos_noise_spectrum = self.cos_noise_temp * np.ones(self.fft_length)
        return self.noise_spectrum, self.uncorrelated_reflected_noise_spectrum, self.sin_noise_spectrum, self.cos_noise_spectrum



class SimulationParameters:
    def __init__(self, number_of_qs, fractional_measurements, time_per_q,
                 receiver):
        self.number_of_qs = number_of_qs
        self.fft_length = receiver.fft_length
        self.time_per_q = time_per_q
        self.receiver_sample_rate = receiver.sample_rate

        self.fraction_on_measurement = fractional_measurements[0]
        self.fraction_on_load1 = fractional_measurements[1]
        self.fraction_on_load2 = fractional_measurements[2]

        self.total_time = self.number_of_qs * self.time_per_q

        self.number_of_reciever_spectrum_samples =  int(self.total_time*self.receiver_sample_rate / self.fft_length)
        self.receiver_spectrum_sample_rate = self.number_of_reciever_spectrum_samples / self.total_time

        self.load1_reciever_samples_per_q = int(self.fraction_on_load1 * self.number_of_reciever_spectrum_samples / self.number_of_qs)
        self.load2_reciever_samples_per_q = int(self.fraction_on_load2 * self.number_of_reciever_spectrum_samples / self.number_of_qs)
        self.measurement_samples_per_q = int(self.fraction_on_measurement * self.number_of_reciever_spectrum_samples / self.number_of_qs)

        self.channel_bandpass = self.receiver_sample_rate / self.fft_length
        self.reciever_spectrum_integration_time = 1 / self.receiver_spectrum_sample_rate
        
        pass

class TimeOrderedDataGenerator:
    def __init__(self, colder_load, hotter_load, to_measure_port,
                 reciever, cw_source, power_meter, simulation_parameters, start_datetime=datetime.datetime.now()):
        self.cw_amplitude_fluctuations = one_over_f_timestream(cw_source.pink_noise_variance,
                                                               cw_source.alpha, 
                                                               simulation_parameters.receiver_spectrum_sample_rate, 
                                                               simulation_parameters.number_of_reciever_spectrum_samples) + 1
        
        self.power_meter_fluctuations = one_over_f_timestream(power_meter.pink_noise_variance, power_meter.alpha, 
                                                              simulation_parameters.receiver_spectrum_sample_rate, 
                                                              simulation_parameters.number_of_reciever_spectrum_samples) + 1
        
        self.reciever_gains_fluctuations = one_over_f_timestream(reciever.pink_noise_variance, reciever.alpha, 
                                                                 simulation_parameters.receiver_spectrum_sample_rate,
                                                                 simulation_parameters.number_of_reciever_spectrum_samples) + 1
        
        self.number_of_sample_per_power_meter_intgration = int(simulation_parameters.receiver_spectrum_sample_rate / power_meter.sample_rate)

        self.switch = int(0) # switch position between loads and source
        self.switch_counter = int(0)
        

        self.cw_spectrum_static = cw_source.brightness_temperature(reciever)

        self.reciever_noise, self.uncorelated_noise, self.sin_noise, self.cos_noise = reciever.reciever_nw_spectrum()

        self.output_spectra = []
        self.cw_powers = []  #at the moment is the actual cw power with self induced fluctuations not from receiver
        self.power_meter_measurements_bits_times = []

        self.time_array = []
        self.time = start_datetime
        self.simulation_interval = datetime.timedelta(seconds=simulation_parameters.reciever_spectrum_integration_time)

        self.switch_time_array = []
        self.switch_time_array.append([self.switch, self.time])

        self.switch_limits = [simulation_parameters.load1_reciever_samples_per_q, simulation_parameters.load2_reciever_samples_per_q,
                              simulation_parameters.measurement_samples_per_q]
        
        self.switch_limits = [simulation_parameters.measurement_samples_per_q, simulation_parameters.load1_reciever_samples_per_q,
                              simulation_parameters.load2_reciever_samples_per_q]
        

        for i in np.arange(simulation_parameters.number_of_reciever_spectrum_samples -1):
            self.cw_term = (self.cw_amplitude_fluctuations[i]) * self.cw_spectrum_static
            self.cw_powers.append((self.cw_amplitude_fluctuations[i])*cw_source.cw_temperature)
            self.time_array.append(self.time)
            

            self.power_meter_measurement = power_meter.output_voltage_bits((self.cw_amplitude_fluctuations[i])*cw_source.cw_power_watts, 
                                                                           self.power_meter_fluctuations[i])
            

            if self.switch == 0:
                self.source_temperatures = colder_load.brightness_temperature(reciever)
                self.source_reflection_coefficints = colder_load.compute_reflection_coefficients(reciever)
            elif self.switch == 1:
                self.source_temperatures = hotter_load.brightness_temperature(reciever)
                self.source_reflection_coefficints = hotter_load.compute_reflection_coefficients(reciever)
            elif self.switch == 2:
                self.source_temperatures = to_measure_port.brightness_temperature(reciever)
                self.source_reflection_coefficints = to_measure_port.compute_reflection_coefficients(reciever)

            if self.switch == 0:
                self.source_temperatures = to_measure_port.brightness_temperature(reciever)
                self.source_reflection_coefficints = to_measure_port.compute_reflection_coefficients(reciever)
            elif self.switch == 1:
                self.source_temperatures = colder_load.brightness_temperature(reciever)
                self.source_reflection_coefficints = colder_load.compute_reflection_coefficients(reciever)
            elif self.switch == 2:
                self.source_temperatures = hotter_load.brightness_temperature(reciever)
                self.source_reflection_coefficints = hotter_load.compute_reflection_coefficients(reciever)
            
            

            
            self.admitance = np.sqrt(1 - np.abs(reciever.reflection_coefficients)**2) / (1 - reciever.reflection_coefficients*self.source_reflection_coefficints)

            self.source_term = self.source_temperatures * (self.admitance**2) * (1-np.abs(self.source_reflection_coefficints)**2)
            self.uncorelated_term = self.uncorelated_noise * (self.admitance**2) * (self.source_reflection_coefficints**2)
            self.sin_term = np.abs(self.admitance)*np.abs(self.source_reflection_coefficints) * self.sin_noise * np.sin(np.angle(self.admitance*self.source_reflection_coefficints))
            self.cos_term = np.abs(self.admitance)*np.abs(self.source_reflection_coefficints) * self.cos_noise * np.cos(np.angle(self.admitance*self.source_reflection_coefficints))
            self.combined_noise_terms = self.source_term + self.uncorelated_term + self.cos_term + self.sin_term + self.reciever_noise

            
            ###
            self.std_array = self.combined_noise_terms / np.sqrt(simulation_parameters.channel_bandpass * simulation_parameters.reciever_spectrum_integration_time)
            self.combined_noise_terms = np.random.normal(loc=np.abs(self.combined_noise_terms),
                                                         scale=np.abs(self.std_array))
            self.combined_terms = self.combined_noise_terms + self.cw_term
            self.spectrum = (self.reciever_gains_fluctuations[i]) * self.combined_terms


            self.output_spectra.append(self.spectrum)
            self.power_meter_measurements_bits_times.append([self.power_meter_measurement, self.time])

            self.switch_counter += 1
            self.time += self.simulation_interval

            if self.switch_counter > self.switch_limits[self.switch]:
                self.switch_counter = 0
                self.switch += 1
                if self.switch >= 3:
                    self.switch = 0
                else:
                    pass
                self.switch_time_array.append([self.switch, self.time])
            else:
                pass
        self.time_array = np.array(self.time_array)
        self.switch_time_array = np.array(self.switch_time_array)
        self.output_spectra = np.array(self.output_spectra)

        self.cw_powers = np.array(self.cw_powers)
        self.power_meter_measurements_bits_times = self.power_meter_measurements_bits_times[::int(reciever.sample_rate / (power_meter.sample_rate * reciever.fft_length))]
        self.power_meter_measurements_bits_times = np.array(self.power_meter_measurements_bits_times)

        self.reciever_gains = self.reciever_gains_fluctuations
        
        pass

    def export_time_ordered_data(self, sample_integration_time, simulation_parameters, file_location, save_to_drive=True):
        self.t_0 = self.time_array[0]
        self.t_int = datetime.timedelta(seconds=sample_integration_time)
        self.end_time = self.t_0 + datetime.timedelta(seconds=simulation_parameters.total_time)
        self.integrated_spectra_to_save = []
        self.power_meter_measurements_to_save = []
        self.datetime_to_save = []

        self.gains_to_save_param = []
        #remember to save the switch states and times
        self.integrate_data = True
        while self.integrate_data:
            self.spectrum_int_array_indices = np.where((self.time_array >= self.t_0) & (self.time_array < self.t_0+self.t_int))
            print(len(self.spectrum_int_array_indices[0]))
            if len(self.spectrum_int_array_indices[0]) == 0:
                pass
            elif len(self.spectrum_int_array_indices[0]) != 0:
                self.spectra_to_integrate = self.output_spectra[self.spectrum_int_array_indices]
            

                self.gains_to_integrate = self.reciever_gains_fluctuations[self.spectrum_int_array_indices]
                self.gains_to_save_param.append(np.mean(self.gains_to_integrate))

                self.integrated_spectra = np.average(self.spectra_to_integrate, axis=0)
                self.integrated_spectra_to_save.append(self.integrated_spectra)

                self.power_meter_int_array_indices = np.where((self.t_0 >= self.power_meter_measurements_bits_times[:,1]) & 
                                                          (self.power_meter_measurements_bits_times[:,1] < self.t_0+self.t_int))
                self.integrated_power_meter_bits = np.mean(self.power_meter_measurements_bits_times[:,0][self.power_meter_int_array_indices])
                self.power_meter_measurements_to_save.append(self.integrated_power_meter_bits)
                self.datetime_to_save.append(self.t_0 + self.t_int/2)
            
            self.t_0 += self.t_int

            if self.t_0 >= self.end_time:
                self.integrate_data = False
            else:
                pass
            pass
        
        self.integrated_spectra_to_save = np.array(self.integrated_spectra_to_save)
        self.power_meter_measurements_to_save = np.array(self.power_meter_measurements_to_save)
        self.datetime_to_save = np.array(self.datetime_to_save)
        self.switch_time_array = np.array(self.switch_time_array)
        self.gains_to_save_param = np.array(self.gains_to_save_param)

        self.frequencies_channels = np.linspace(reciever.centre_frequency - reciever.sample_rate/2, 
                                                reciever.centre_frequency + reciever.sample_rate/2, reciever.fft_length)

        if save_to_drive:
            np.save(file_location+'Spectra', self.integrated_spectra_to_save)
            np.save(file_location+'PowerMeterMeasurements', self.power_meter_measurements_to_save)
            np.save(file_location+'DateTimes', self.datetime_to_save)
            np.save(file_location+'SwitchTimes', self.switch_time_array)
            np.save(file_location+'ChannelFrequencies', self.frequencies_channels)
            np.save(file_location+'param_recgains.npy', self.gains_to_save_param)
            print('Ding Ding: saved under: '+file_location)
        else:
            return self.integrated_spectra_to_save, self.power_meter_measurements_to_save, self.datetime_to_save, self.switch_time_array, self.frequencies_channels


    
    def plot_water_fall(self, receiver, simulation_parameters, saveplot=False, savepath=''):
        extent = [(receiver.centre_frequency + receiver.sample_rate/-2)/1e6,
                  (receiver.centre_frequency + receiver.sample_rate/2)/1e6,
                  simulation_parameters.total_time, 0]
        plt.imshow(10*np.log10(np.abs(self.output_spectra)), aspect='auto', extent=extent)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.colorbar()
        plt.tight_layout()
        if saveplot:
            plt.savefig(savepath+'waterfall.pdf')
        plt.show()
    
    def plot_power_meter_bits(self):
        plt.plot(self.power_meter_measurements_bits_times[:,1], self.power_meter_measurements_bits_times[:,0])
        plt.ylabel('Power meter measurement [level]')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()

    def plot_cw_brightness(self, simulation_parameters):
        plt.plot(np.linspace(0, simulation_parameters.total_time, len(self.cw_powers)), self.cw_powers)
        plt.ylabel('CW Power [arb]')
        plt.xlabel('Time [s]')
        plt.tight_layout
        plt.show()

    def plot_gain_fluctuations(self, simulation_parameters):
        plt.plot(np.linspace(0, simulation_parameters.total_time, len(self.reciever_gains)), self.reciever_gains)
        plt.ylabel('Gain fluctuations [arb]')
        plt.xlabel('Time [s]')
        plt.tight_layout
        plt.show()
    
class DataProcessor:
    def __init__(self, file_location, number_of_side_cw_channels, number_of_cw_sys_channels, power_meter, cold_load, hot_load, no_switching=False, include_power_meter=False):
        print('Loading and processing Data')

        self.no_switching = no_switching
        self.file_location = file_location

        self.switch_times = np.load(self.file_location+'SwitchTimes.npy', allow_pickle=True) # 0 -measure, 1-cold, 2-hot
        self.waterfall_spectra = np.load(self.file_location+'Spectra.npy')
        self.power_meter_measurements = np.load(self.file_location+'PowerMeterMeasurements.npy')
        self.datetimes = np.load(self.file_location+'DateTimes.npy', allow_pickle=True)
        self.channel_frequencies = np.load(self.file_location+'ChannelFrequencies.npy')

        self.cw_side_channel_number = number_of_side_cw_channels
        self.cw_side_sys_number = number_of_cw_sys_channels

        self.gain_calibrated_spectra_array = []
        self.cw_power_measurements = []
        #perform calibration for each sample
        self.switch_buffer_time = (self.datetimes[-1] - self.datetimes[0]) / len(self.datetimes)
        self.power_meter_baseline_power = power_meter.convert_bit_voltgage_to_power(power_meter.output_voltage_bits(0, 1))

        for i in np.arange(len(self.datetimes)):
            self.receiver_cw = cw_isolation_channel_difference(self.waterfall_spectra[i], 
                                                               self.cw_side_channel_number,
                                                               self.cw_side_sys_number)
            self.power_meter_value = power_meter.convert_bit_voltgage_to_power(self.power_meter_measurements[i]) - self.power_meter_baseline_power
            self.spectra = self.waterfall_spectra[i]

            if include_power_meter:
                self.gain_calibrated_spectra = self.spectra * self.power_meter_value / self.receiver_cw
            else:
                self.gain_calibrated_spectra = self.spectra / self.receiver_cw

            self.gain_calibrated_spectra_array.append(self.gain_calibrated_spectra)
            self.cw_power_measurements.append(self.receiver_cw)

        self.cw_power_measurements = np.array(self.cw_power_measurements)
        self.gain_calibrated_spectra_array = np.array(self.gain_calibrated_spectra_array)

        self.integrated_spectra_per_switch = []
        self.integrated_calibrated_spectra_switch_state = []
        for i in np.arange(len(self.switch_times[:,1])):
            if i == len(self.switch_times[:,1]) -1:
                self.calibrated_spectra_to_average_indices = np.where(self.datetimes >= self.switch_times[:,1][i] + self.switch_buffer_time)
                if len(self.calibrated_spectra_to_average_indices) == 0:
                    
                    pass
                else:
                    self.integrated_calibrated_spectra = np.mean(self.gain_calibrated_spectra_array[self.calibrated_spectra_to_average_indices], axis=0)
                    self.integrated_spectra_per_switch.append(self.integrated_calibrated_spectra)
                    self.integrated_calibrated_spectra_switch_state.append(self.switch_times[:,0][i])
            else:
                self.calibrated_spectra_to_average_indices = np.where((self.datetimes > self.switch_times[:,1][i] + self.switch_buffer_time) &
                                                                      (self.datetimes < self.switch_times[:,1][i+1] - self.switch_buffer_time))
                self.integrated_calibrated_spectra = np.mean(self.gain_calibrated_spectra_array[self.calibrated_spectra_to_average_indices], axis=0)
                self.integrated_spectra_per_switch.append(self.integrated_calibrated_spectra)
                self.integrated_calibrated_spectra_switch_state.append(self.switch_times[:,0][i])
        
        self.integrated_spectra_per_switch = np.array(self.integrated_spectra_per_switch)
        self.integrated_calibrated_spectra_switch_state = np.array(self.integrated_calibrated_spectra_switch_state)
        
        if no_switching:
            return

        self.q_array = []
        for j in np.arange(np.floor(len(self.integrated_spectra_per_switch) / 3)):
            self.indices_of_interest = list(range(int(j*3), int(3*(j+1))))
            self.powers_of_interest = self.integrated_spectra_per_switch[self.indices_of_interest]
            self.switch_state_of_interest = self.integrated_calibrated_spectra_switch_state[self.indices_of_interest]

            self.cold_load = self.powers_of_interest[np.where(self.switch_state_of_interest == 1)]
            self.hot_load = self.powers_of_interest[np.where(self.switch_state_of_interest == 2)]
            self.measurement_port = self.powers_of_interest[np.where(self.switch_state_of_interest == 0)]

            self.q = (self.measurement_port - self.cold_load) / (self.hot_load - self.cold_load)
            
            self.q_array.append(self.q[0])

        self.t_meas_array = []
        for q in self.q_array:
            t_meas = q*(hot_load.physical_temperature - cold_load.physical_temperature) + cold_load.physical_temperature
            self.t_meas_array.append(t_meas)
        
        self.t_meas_array = np.array(self.t_meas_array)
        
        self.measured_temperature_at_port = np.mean(self.t_meas_array, axis=0)  # measured calibrated noise temperature at reciever port including noise wave contributions
        print('Data Processed')
       
        pass


    def plot_t_measured_spectrum(self, save_plot=False, savepath=''):
        if self.no_switching:
            return
        self.spectrum_standard_deviation = np.std(self.measured_temperature_at_port)
        self.spectrum_mean = np.mean(self.measured_temperature_at_port)
        
        plt.plot(self.channel_frequencies / 10**6, np.abs(self.measured_temperature_at_port))
        #plt.ylim(0, 500)
        plt.xlabel(r'$\nu$ [MHz]')
        plt.ylabel('Calibrated Noise Temperature [K]')
        plt.grid()
        plt.tight_layout()
        if save_plot:
            plt.savefig(savepath+'Cal_Spectrum.pdf')
        plt.show()
        pass

    def plot_waterfall(self, save_plot=False, savepath=''):
        extent = [self.channel_frequencies[0] /1e6, self.channel_frequencies[-1]/ 1e6,
                   self.datetimes[-1], self.datetimes[0]]
        plt.imshow(10*np.log10(self.waterfall_spectra), aspect='auto', extent=extent)
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [LST]")
        plt.colorbar(label='Measured Channel Power [dB]')
        plt.tight_layout()
        if save_plot:
            plt.savefig(savepath+'waterfall.pdf')
        plt.show()

    def plot_post_detection_psd(self, frequency_mhz):
        index = int(frequency_mhz * len(self.channel_frequencies) / (self.channel_frequencies[-1] - self.channel_frequencies[0]))
        powers = self.waterfall_spectra[:,index]
        powers = powers / np.mean(powers)
        run_time_seconds = (self.datetimes[-1] - self.datetimes[0]).total_seconds()
        gain_calibrated_powers = self.gain_calibrated_spectra_array[:,index]
        gain_calibrated_powers = gain_calibrated_powers / np.mean(gain_calibrated_powers)
        psd = np.abs(np.fft.rfft(powers, len(powers)))
        gain_cal_psd = np.abs(np.fft.rfft(gain_calibrated_powers, len(gain_calibrated_powers)))
        post_detection_frequencies = np.fft.rfftfreq(len(powers), d=run_time_seconds/len(powers))
        gain_cal_post_detection_frequencies = np.fft.rfftfreq(len(gain_calibrated_powers), d=run_time_seconds/len(gain_calibrated_powers))

        plt.plot(post_detection_frequencies, psd, label='Uncorrected', c='Black')
        plt.plot(gain_cal_post_detection_frequencies, gain_cal_psd, label='Gain Corrected', c='Blue')
        
        plt.xlabel('Post Detection Frequency [Hz]')
        plt.ylabel('PSD of normalised deviations [arb.]')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.show()

        plt.plot(self.datetimes, powers, label='Uncorrected')
        plt.plot(self.datetimes, gain_calibrated_powers, label='Gain Corrected')
        plt.ylabel('Normalised Power')
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_cw_powers_and_gains(self, saveplot=False, savepath=''):
        gains = np.load(self.file_location + 'param_recgains.npy')
        gains = gains / np.mean(gains)
        cw_measurements = self.cw_power_measurements
        times = self.datetimes
        cw_measurements = cw_measurements / np.mean(cw_measurements)
        plt.plot(times, cw_measurements, label='cw_measurement')
        plt.plot(times, gains, label='simulated_gains')
        plt.ylabel('Normalised Fluctuations')
        plt.legend()
        plt.tight_layout()
        if saveplot:
            plt.savefig(savepath+'cwAndgains.pdf')
        plt.show()

        

if __name__ == "__main__":
    cold_load = Load(physical_temperature=383)

    hot_load = Load(physical_temperature=2000)

    reference_load = Load(physical_temperature=500)

    open_cable = OpenCable(physical_temperature=310, two_way_delay=50*10**-9, two_way_cable_loss_db=1)

    reciever = Reciever(sample_rate=20*10**6, centre_frequency=70*10**6, fft_length=2048, noise_temperature=150,
                        uncorrelated_reflected_noise_temp=60, sin_noise_temp=30, cos_noise_temp=-20, pink_noise_variance=10,
                        alpha=2, window='Blackman', reflection_coefficients=0.1)

    cw_source = CWSource(pink_noise_variance=0.5, alpha=2, cw_power_dbm=-15, cw_frequency=75*10**6, attenuation_before_receiver_db=-40 ,receiver=reciever)

    log_power_meter = PowerMeter(feedback_factor_x=2, pink_noise_variance=1e-9, sample_rate=12, alpha=1.5, bit_depth=12, 
                             physical_temperature=310, bandwidth=2*10**9)

    simulation_parameters = SimulationParameters(number_of_qs=4, fractional_measurements=[1/3,1/3,1/3], 
                                                 time_per_q=30, receiver=reciever)

    #tod = TimeOrderedDataGenerator(cold_load, hot_load, reference_load, reciever, cw_source, log_power_meter, simulation_parameters)
    #tod.export_time_ordered_data(1/log_power_meter.sample_rate, simulation_parameters=simulation_parameters, file_location='ExportPlots_2/')
    
    #generate_multiple_qs(time_per_q=30, number_of_qs=5, cold_load=cold_load, hot_load=hot_load, 
    #                     to_meausure=reference_load, receiver=reciever, 
    #                     cw_source=cw_source, power_meter=log_power_meter,
    #                     fractional_measurements=[1/3,1/3,1/3], file_location='ExportPlots_2/')

    processed_spectra = DataProcessor('ExportPlots_2/', number_of_cw_sys_channels=2, number_of_side_cw_channels=3, 
                                      power_meter=log_power_meter, cold_load=cold_load, hot_load=hot_load, no_switching=False)
    


    processed_spectra.plot_waterfall(save_plot=True, savepath='ExportPlots_2/')
    processed_spectra.plot_t_measured_spectrum(save_plot=True, savepath='ExportPlots_2/')

#tod.plot_cw_brightness(simulation_parameters)
#tod.plot_gain_fluctuations(simulation_parameters)
#tod.plot_power_meter_bits()

    #processed_spectra.plot_t_measured_spectrum()

    #processed_spectra.plot_post_detection_psd(67)
    processed_spectra.plot_cw_powers_and_gains(saveplot=True, savepath='ExportPlots_2/')
