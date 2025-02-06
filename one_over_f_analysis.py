import numpy as np
import matplotlib.pyplot as plt
import sdr_read_in
from scipy.optimize import curve_fit

def compute_psd_from_array(arr, delta_t):
    psd = np.sqrt(delta_t/len(arr)) * np.abs(np.fft.rfft(arr))
    return psd

def psd_model(freq, white_noise_variance, f_knee, alpha):
    psd = white_noise_variance*(1+ (f_knee/freq)**alpha)
    return psd

def measure_cw_power_with_sys_noise(spectra):
    cw_index = np.argmax(spectra)
    cw_power = np.sum(spectra[cw_index-2:cw_index+2])
    return cw_power

def measure_cw_power_diff(spectra):
    cw_index = np.argmax(spectra)
    cw_power = np.sum(spectra[cw_index-2:cw_index+2])
    upper_background = np.sum(spectra[cw_index+5-2:cw_index+5+2])
    lower_background = np.sum(spectra[cw_index-5-2:cw_index-5+2])
    isolated_cw = cw_power - np.mean([upper_background, lower_background])
    return isolated_cw

def average_waterfall_to_timescale(waterfall, times, averaging_time):
    t_0 = times[0]
    averaged_spectra = []
    new_times = []
    run=True
    while run:
        spectra_to_average_indices = np.where((times >= t_0) & (times < t_0+averaging_time))
        spectra_to_average = waterfall[spectra_to_average_indices]
        avg_spec = np.mean(spectra_to_average, axis=0)
        averaged_spectra.append(avg_spec)
        new_times.append(t_0 + averaging_time/2)
        t_0 += averaging_time
        if t_0 >= times[-1]:
            run=False
    return np.array(averaged_spectra), np.array(new_times)

class OneOverFAnalyser:
    def __init__(self, file_root, filename=''):
        self.root_path = file_root
        self.filename = filename
        pass

    def compute_psd_for_cw(self, save=False, savepath='', isolate_cw=False, average_to_time=False, average_to_time_s=1):
        waterfall, times, centre_freq, sample_rate = sdr_read_in.SDRReader(self.root_path).read_from_csv_and_txt(self.filename)
        del centre_freq, sample_rate
        if average_to_time:
            waterfall, times = average_waterfall_to_timescale(waterfall, times, average_to_time_s)
        cw_powers = []
        for spectra in waterfall:
            if isolate_cw:
                cw_powers.append(measure_cw_power_diff(spectra))
            else:
                cw_powers.append(measure_cw_power_with_sys_noise(spectra))
        del waterfall
        cw_powers = np.array(cw_powers)
        delta_t = (times[-1] - times[0]) / len(times)
        self.mean_psd = compute_psd_from_array(cw_powers, delta_t=delta_t)
        self.freqs_psd = np.fft.rfftfreq(len(times), d=(times[-1] - times[0])/len(times))
        self.std_psd = self.mean_psd / np.sqrt(len(cw_powers))

        if save:
            np.save(savepath+self.filename+'_meanPsd.npy', self.mean_psd)
            np.save(savepath+self.filename+'_StdPsd.npy', self.std_psd)
            np.save(savepath+self.filename+'_FreqsPsd.npy', self.freqs_psd)

        return self.mean_psd, self.std_psd, self.freqs_psd


    def compute_all_temporal_psd_from_file(self, save=False, savepath=''):
        waterfall, times, centre_freq, sample_rate = sdr_read_in.SDRReader(self.root_path).read_from_csv_and_txt(self.filename)
        del centre_freq,sample_rate
        f_channel_num = len(waterfall[0])
        delta_t = (times[-1] - times[0]) / len(times)
        psd_over_freq = []

        for i in np.arange(len(waterfall[0])):
            psd_over_freq.append(compute_psd_from_array(waterfall[:,i], delta_t=delta_t))
        del waterfall
        psd_over_freq = np.array(psd_over_freq)

        self.mean_psd = np.mean(psd_over_freq, axis=0)
        self.std_psd = np.std(psd_over_freq, axis=0) / np.sqrt(f_channel_num)
        self.freqs_psd = np.fft.rfftfreq(len(times), d=(times[-1] - times[0]) / len(times) )
        if save:
            np.save(savepath+self.filename+'_meanPsd.npy', self.mean_psd)
            np.save(savepath+self.filename+'_StdPsd.npy', self.std_psd)
            np.save(savepath+self.filename+'_FreqsPsd.npy', self.freqs_psd)



        return self.mean_psd, self.std_psd, self.freqs_psd
    
    def load_psd(self, path):
        self.mean_psd = np.load(path+self.filename+'_meanPsd.npy')
        self.std_psd = np.load(path+self.filename+'_StdPsd.npy')
        self.freqs_psd = np.load(path+self.filename+'_FreqsPsd.npy')
    
    def plot_psd_errror(self, save=False, save_path="Filepath"):
        plt.errorbar(self.freqs_psd[1:], self.mean_psd[1:], self.std_psd[1:], fmt='o')
        plt.ylabel('PSD')
        plt.xlabel('Temporal Frequency [Hz]')
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        if save:
            plt.savefig(save_path+self.filename+'_psd_err.pdf')
        
        plt.show()
        pass
    def plot_psd(self, save=False, save_path="Filepath"):
        plt.plot(self.freqs_psd[1:], self.mean_psd[1:])
        plt.ylabel('PSD')
        plt.xlabel('Temporal Frequency [Hz]')
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        if save:
            plt.savefig(save_path+self.filename+'_psd.pdf')
        
        plt.show()
        pass


    def fit_for_psd_params(self, no_plot=False, save_image=False, image_path=''):
        popt, pcov = curve_fit(psd_model, self.freqs_psd[2:], self.mean_psd[2:], sigma=self.std_psd[2:], nan_policy='omit')
        print('--fitted--')
        print('White Noise Variance: '+str(popt[0])+' +- '+str(np.sqrt(pcov[0,0])))
        print('f_knee: '+str(popt[1])+' +- '+str(np.sqrt(pcov[1,1])))
        print('Alpha: '+str(popt[2])+' +- '+str(np.sqrt(pcov[2,2])))

        print(pcov)

        if no_plot:
            return

        plt.plot(self.freqs_psd[1:], psd_model(self.freqs_psd[1:], popt[0], popt[1], popt[2]), label='Model')
        plt.plot(self.freqs_psd[1:], self.mean_psd[1:], label='Data')
        #plt.hlines(popt[0], xmin=self.freqs_psd[0], xmax=self.freqs_psd[-1], colors='green',
        #           linestyles='--')
        slope = lambda f,f_k, wn,alpha: wn*((f_k/f)**alpha )
        #plt.plot(self.freqs_psd[1:20], slope(self.freqs_psd[1:20], popt[1], wn=popt[0],alpha=popt[2]))
        plt.ylabel('PSD')
        plt.xlabel('Temporal Frequency [Hz]')
        plt.yscale('log')
        plt.xscale('log')
        #plt.grid()
        plt.tight_layout()
        plt.legend()
        if save_image:
            plt.savefig(image_path+self.filename+'_psd_fit.pdf')
        
        plt.show()

        pass


if __name__ == "__main__":
    #analyser = OneOverFAnalyser('Data/', filename="2025_01_16_104232")
    #analyser.compute_all_temporal_psd_from_file(save=True, savepath='SavedPSDs/')
    #analyser.compute_psd_for_cw(save=True, savepath='SavedPSDs/', isolate_cw=False, average_to_time=True, average_to_time_s=0.1)
    #analyser.load_psd('SavedPSDs/')
    #analyser.plot_psd(save=True, save_path='ImagePath/')
    #analyser.plot_psd_errror(save=True, save_path='ImagePath/')
    #analyser.fit_for_psd_params(no_plot=False, save_image=True, image_path='ImagePath/')

    an1 = OneOverFAnalyser('Data/', filename="2025_01_16_104232")
    an1.compute_psd_for_cw(save=True, savepath='SavedPSDs/', isolate_cw=False, average_to_time=True, average_to_time_s=0.1)

    an2 = OneOverFAnalyser('Data/', filename="2025_01_16_110340")
    an2.compute_psd_for_cw(save=True, savepath='SavedPSDs/', isolate_cw=False, average_to_time=True, average_to_time_s=0.1)

    an3 = OneOverFAnalyser('Data/', filename="2025_01_16_113134")
    an3.compute_psd_for_cw(save=True, savepath='SavedPSDs/', isolate_cw=False, average_to_time=True, average_to_time_s=0.1) 
    pass