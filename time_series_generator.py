import charger
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    f_cw = [0.01, 0.1, 1, 10, 100]
    pm_white_noises = [0.01, 0.1, 1, 10, 100]

    receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                    n_freq_channels=2**13, sample_rate=20e6,
                                    t_unc=250, t_cos=190, t_sin=90, t_n=300,
                                    reflection_coefficients=0.5, window_function='Rectangular',
                                    beta=0)
    #target = charger.TerminatedCable(physical_temperature=300, frequencies=receiver.frequencies, termination='Open',
    #                                 epsilon=1.5, cable_length=30)
    
    load = charger.Source(temperatures=373, reflection_coefficients=0, frequencies=receiver.frequencies)
    #noise_source = charger.Source(temperatures=1200, reflection_coefficients=0, frequencies=receiver.frequencies)

    cw_sources = [charger.CW_Source(initial_cw_amplitude=100, oscilator_frequency=72e6, characteristic_frequency=receiver.characteristic_frequency*f, alpha=2, 
                                    phase_white_noise=1e-9, phase_knee_frequency=1e-9, phase_alpha=0) for f in f_cw]
    
    power_meter = charger.BasicPowerMeter(5, 3, characteristic_frequency=0.01*receiver.characteristic_frequency, alpha=2, sample_rate=13, white_noise_level=1)
    

    directories = ['Test_Data/PowerMeterTests/wn0.01/',
                   'Test_Data/PowerMeterTests/wn0.1/',
                   'Test_Data/PowerMeterTests/wn1/',
                   'Test_Data/PowerMeterTests/wn10/',
                   'Test_Data/PowerMeterTests/wn100/']
    
    for pmwn, d in zip(pm_white_noises, directories):
        for f in f_cw:
            receiver = charger.SDR_Receiver(characteristic_frequency=100, alpha=2, centre_frequency=70e6,
                                    n_freq_channels=2**13, sample_rate=20e6,
                                    t_unc=250, t_cos=190, t_sin=90, t_n=300,
                                    reflection_coefficients=0.5, window_function='Rectangular',
                                    beta=0)
            load = charger.Source(temperatures=300, reflection_coefficients=0, frequencies=receiver.frequencies)

            cw_source = charger.CW_Source(initial_cw_amplitude=100, oscilator_frequency=72e6, characteristic_frequency=receiver.characteristic_frequency*f, alpha=2, 
                                    phase_white_noise=1e-9, phase_knee_frequency=1e-9, phase_alpha=0)
            
            power_meter = charger.BasicPowerMeter(5, 3, characteristic_frequency=0.01*receiver.characteristic_frequency, alpha=2, sample_rate=13, white_noise_level=pmwn)

            t = charger.TimeStreamGenerator(integration_time=1, simulation_time=600, #change to 600
                                        bandwidth=receiver.sample_rate,
                                        centre_frequency=receiver.centre_frequency,
                                        n_freq_channels=receiver.n_freq_channels)
            title = f'tod_cw_fk_{f}.hd5f'
            t.generate_simulated_data(obs_source=load, cw_source=cw_source, receiver=receiver,
                                  save_data=True, savepath=d, title=title, save_into_object=True, plot_spectra=False,
                                  switching=False, power_meter=power_meter)


    print('-----------===========---------')
    print('             All Done')
    print('-----------===========---------')

    
