import charger
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_windowing_tests():
    windows = ['Rectangular','Blackman', 'Bartlett', 'Hamming', 'Hanning', 'BlackmanHarris', 'Cosine']
    
    receiver_list = [charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                          n_freq_channels=2**13, sample_rate=20e6,
                                          t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function=window, beta=0) for window in windows]

    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=receiver_list[0].frequencies)

    cw_amps = [0.01, 1, 100]

    cw_sources = [charger.CW_Source(initial_cw_amplitude=a, oscilator_frequency=75e6, characteristic_frequency=receiver_list[0].characteristic_frequency*0.01,
                                   alpha=2) for a in cw_amps]
    
    power_meter = charger.BasicPowerMeter(5, 3, characteristic_frequency=cw_sources[0].characteristic_frequency*0.01, alpha=2, sample_rate=13, white_noise_level=1)

    # Set up folder for this run if it doesn't exist

    save_path = 'Generated_Data/WindowTests'

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = 'Generated_Data/WindowTests/'

    for window, rec in zip(windows, receiver_list):
        for cw_amp, cw_source in zip(cw_amps, cw_sources):
            t = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, bandwidth=rec.sample_rate,
                                            centre_frequency=rec.centre_frequency, n_freq_channels=rec.n_freq_channels)
            title = f'win{window}_cwamp{str(cw_amp)}_test.hd5f'

            t.generate_simulated_data(obs_source=load, cw_source=cw_source,
                                      receiver=rec, save_data=True, savepath=save_path,title=title,
                                      save_into_object=False, plot_spectra=False,
                                      switching=False, power_meter=power_meter)
            print(f' -- Test Window = {window}')
            print(f' cw_amp = {cw_amp}')
            print(f'--Done--')

    print('======================================')
    print('Window CW Amp Data Generation Finished')
    print('======================================')
    return



def test_noise_wave_extraction():
    save_path = 'Generated_Data/NoiseWaveExtraction'

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = 'Generated_Data/NoiseWaveExtraction/'
    # Test for inclusion of no CW signal and with CW signal

    
    receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                    n_freq_channels=2**13, sample_rate=20e6,
                                    t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function='Blackman', beta=0)
    
    cw_source = charger.CW_Source(initial_cw_amplitude=1, oscilator_frequency=72e6, characteristic_frequency=receiver.characteristic_frequency*0.01, alpha=2)

    noise_diode = charger.Source(temperatures=5000, reflection_coefficients=0, frequencies=receiver.frequencies)
    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=receiver.frequencies) # Set up the load and noise_diode

    # Set up 8 noise wave calibrators

    cal_1 = charger.TerminatedCable(physical_temperature=300, frequencies=receiver.frequencies, termination='Short', epsilon=1,
                                    cable_length=1, mag_s12=1)
    cal_2 = charger.TerminatedCable(physical_temperature=300, frequencies=receiver.frequencies, termination='Open', epsilon=1,
                                    cable_length=1, mag_s12=1)
    
    cal_3 = charger.TerminatedCable(physical_temperature=300, frequencies=receiver.frequencies, termination='Open', epsilon=1,
                                    cable_length=10, mag_s12=0.5)
    cal_4 = charger.TerminatedCable(physical_temperature=300, frequencies=receiver.frequencies, termination='Short', epsilon=1,
                                    cable_length=10, mag_s12=0.5)
    
    cal_5 = charger.TerminatedCable(physical_temperature=370, frequencies=receiver.frequencies, termination='Short', epsilon=1,
                                    cable_length=1, mag_s12=1)
    cal_6 = charger.TerminatedCable(physical_temperature=370, frequencies=receiver.frequencies, termination='Open', epsilon=1,
                                    cable_length=1, mag_s12=1)
    
    cal_7 = charger.TerminatedCable(physical_temperature=370, frequencies=receiver.frequencies, termination='Short', epsilon=1,
                                    cable_length=1, mag_s12=0.5)
    cal_8 = charger.TerminatedCable(physical_temperature=370, frequencies=receiver.frequencies, termination='Open', epsilon=1,
                                    cable_length=1, mag_s12=0.5)

    calibrator_list = [cal_1, cal_2, cal_3, cal_4, cal_5, cal_6, cal_7, cal_8]

    for i, calibrator in enumerate(calibrator_list):
        cw_title = f'cw_cal{str(i+1)}.hd5f'
        no_cw_title = f'nocw_cal{str(i+1)}.hd5f'

        t = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, bandwidth=receiver.sample_rate, centre_frequency=receiver.centre_frequency,
                                        n_freq_channels=receiver.n_freq_channels)
        t.generate_simulated_data(obs_source=load, cw_source=cw_source, receiver=receiver, save_data=True, savepath=save_path, title=cw_title,
                                  switching=True, switch_sources=[load, noise_diode, calibrator], switch_cycle_period=60)
        t.generate_simulated_data(obs_source=load, cw_source=None, receiver=receiver, save_data=True, savepath=save_path, title=no_cw_title,
                                  switching=True, switch_sources=[load, noise_diode, calibrator], switch_cycle_period=60)
    print('======================================')
    print('Noise Wave Param Extraction Data Generation Finished')
    print('======================================')
    return




def decorelated_gains_tests():
    windows = ['Rectangular','Blackman', 'BlackmanHarris']
    
    receiver_list = [charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                          n_freq_channels=2**13, sample_rate=20e6,
                                          t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function=window, beta=0) for window in windows]

    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=receiver_list[0].frequencies)

    cw_amps = [0.01, 1, 100]

    betas = [0.001, 0.01, 0.1, 0.5]

    cw_sources = [charger.CW_Source(initial_cw_amplitude=a, oscilator_frequency=75e6, characteristic_frequency=receiver_list[0].characteristic_frequency*0.01,
                                   alpha=2) for a in cw_amps]
    
    power_meter = charger.BasicPowerMeter(5, 3, characteristic_frequency=cw_sources[0].characteristic_frequency*0.01, alpha=2, sample_rate=13, white_noise_level=1)

    # Set up folder for this run if it doesn't exist

    save_path = 'Generated_Data/DecorrelatedGains'

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = 'Generated_Data/DecorrelatedGains/'

    for window in windows:
        for cw_amp, cw_source in zip(cw_amps, cw_sources):
            for beta in betas:

                rec = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6, n_freq_channels=2**13, sample_rate=20e6,
                                            t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function=window, beta=beta)


                t = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, bandwidth=rec.sample_rate,
                                            centre_frequency=rec.centre_frequency, n_freq_channels=rec.n_freq_channels)
                title = f'beta{str(beta)}_win{window}_cwamp{str(cw_amp)}_test.hd5f'

                t.generate_simulated_data_restricted_gains(obs_source=load, cw_source=cw_source,
                                                           receiver=rec, save_data=True, savepath=save_path,title=title,
                                                           switching=False, power_meter=power_meter)
                print(f' -- Test Window = {window}')
                print(f' cw_amp = {cw_amp}')
                print(f'--Done--')

    print('======================================')
    print('Decorrelated Gains Data Generation Finished')
    print('======================================')
    return



def power_meter_params_tests():
    
    save_path = 'Generated_Data/PowerMeterTests'

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = 'Generated_Data/PowerMeterTests/'


    receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                    n_freq_channels=2**13, sample_rate=20e6,
                                    t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function='Blackman', beta=0)
    
    pm_white_noise_levels = [0.0001,
                             0.001,
                             0.01,
                             0.1,
                             1]
    
    cw_amps = [0.01, 0.1,1, 10,100]
    
    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=receiver.frequencies)

    for cw_amp in cw_amps:
        for pm_wn in pm_white_noise_levels:
            power_meter = charger.BasicPowerMeter(10,10, characteristic_frequency=receiver.characteristic_frequency*0.001, alpha=2,
                                                  sample_rate=13, white_noise_level=pm_wn)
            
            unstable_cw = charger.CW_Source(initial_cw_amplitude=cw_amp, oscilator_frequency=75e6, characteristic_frequency=receiver.characteristic_frequency*5,
                                            alpha=2)
    
            stable_cw = charger.CW_Source(initial_cw_amplitude=cw_amp, oscilator_frequency=75e6, characteristic_frequency=receiver.characteristic_frequency*0.01,
                                          alpha=2)
            
            t_unstable = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, bandwidth=receiver.sample_rate, centre_frequency=receiver.centre_frequency,
                                                     n_freq_channels=receiver.n_freq_channels)
            
            t_stable = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, bandwidth=receiver.sample_rate, centre_frequency=receiver.centre_frequency,
                                                     n_freq_channels=receiver.n_freq_channels)
            
            unstable_title = f'unstableCW_cwamp{str(cw_amp)}_pmwn{str(pm_wn)}.hd5f'
            stable_title = f'stableCW_cwamp{str(cw_amp)}_pmwn{str(pm_wn)}.hd5f'

            t_unstable.generate_simulated_data(obs_source=load, cw_source=unstable_cw,receiver=receiver,save_data=True,
                                               savepath=save_path, title=unstable_title, switching=False, plot_spectra=False, power_meter=power_meter)
            t_stable.generate_simulated_data(obs_source=load, cw_source=stable_cw,receiver=receiver,save_data=True,
                                               savepath=save_path, title=stable_title, switching=False, plot_spectra=False, power_meter=power_meter)
            
            print('Saved: '+ unstable_cw)
            print('Saved: '+ stable_title)
    
    print('======================================')
    print('PowerMeter Data Generation Finished')
    print('======================================')
    return


def phase_noise_tests():
    save_path = 'Generated_Data/PhaseNoiseTest'

    if not os.path.exists(path=save_path):
        os.makedirs(save_path)
        print('Save Path Created')
    else:
        print('Save Path Set Up')
    save_path = 'Generated_Data/PhaseNoiseTest/'


    white_phase_noise_level = [1e-8, 1e-6, 1e-4, 1e-2, 1]
    random_walk_phase_levels = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

    windows = ['Blackman', 'BlackmanHarris','Rectangular']   

    mock_receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                    n_freq_channels=2**13, sample_rate=20e6,
                                    t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function='Blackman', beta=0)

    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=mock_receiver.frequencies)

    for wpn in white_phase_noise_level:
        for rwpn in random_walk_phase_levels:
            for window in windows:
                receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                                n_freq_channels=2**13, sample_rate=20e6,
                                                t_unc=250, t_cos=50, t_sin=50, t_n=300, reflection_coefficients=0.5, window_function=window, beta=0)
                cw_source = charger.CW_Source(initial_cw_amplitude=10, oscilator_frequency=75e6, characteristic_frequency=receiver.characteristic_frequency*0.01,
                                   alpha=2, phase_noise_params=[0,0,0,rwpn,wpn]) # add phase noise
                title = f'win{str(window)}_wpn{str(wpn)}_rwpn{rwpn}.hd5f'

                t.generate_simulated_data(obs_source=load, cw_source=cw_source, receiver=receiver, save_data=True, savepath=save_path,
                                          title=title, switching=False, plot_spectra=False, save_into_object=False)

                
    print('======================================')
    print('PhaseNoise Data Generation Finished')
    print('======================================')
    return


if __name__ == '__main__':
    generate_windowing_tests()
    decorelated_gains_tests()
    power_meter_params_tests()
    phase_noise_tests()
    test_noise_wave_extraction()

    print('-----------===========---------')
    print('             All Done')
    print('-----------===========---------')
