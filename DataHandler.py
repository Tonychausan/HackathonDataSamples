import numpy
import pywt

import Utility


EMG_WAVELET_LEVEL = 1


class DataHandler:
    def __init__(self, file):
        self.emg_data = numpy.genfromtxt(file.get_file_path(), delimiter=',')
        
    def get_emg_sums_normalized(self):
        emg_sums = []
        emg_sum_min = -1
        emg_sum_max = -1
        for emg_array in self.emg_data:
            emg_sum = numpy.sum(numpy.square(emg_array))
            emg_sums.append(emg_sum)

            if emg_sum_min == -1:
                emg_sum_min = emg_sum
                emg_sum_max = emg_sum
            elif emg_sum < emg_sum_min:
                emg_sum_min = emg_sum
            elif emg_sum > emg_sum_max:
                emg_sum_max = emg_sum

        emg_sums = Utility.NormalizeArray(emg_sums)
        emg_sums = numpy.append(emg_sums, self.get_waveform_length_of_emg()).flatten()

        return emg_sums

    def wavelet_feature_extraxtion(self):
        # global_max = 0
        # global_min = -1
        # for emg_array in self.emg_data:
        #     local_max = numpy.max(emg_array)
        #     local_min = numpy.min(emg_array)

        #     if global_max < local_max:
        #         global_max = local_max

        #     if global_min == -1 or global_min > local_min:
        #         global_min = local_min

        mav_data = []
        rms_data = []
        wl_data = []

        emg_feature_data = [mav_data, rms_data, wl_data]

        n = EMG_WAVELET_LEVEL
        for emg_array in self.emg_data:
            # emg_array = Utility.NormalizeArray(emg_array)
            # emg_array = numpy.subtract(emg_array, global_min)
            # emg_array = numpy.divide(emg_array, global_max - global_min)

            coeffs = pywt.wavedec(emg_array, 'db1', level=n)
            cAn = coeffs[0]
            cD = coeffs[1:]

            reconstructed_signal = []
            for i in range(n + 1):
                temp_coeffs = [None] * (n + 1 - i)  # placement of [cAn, cDn, cD(n-1)..., cD1]
                if i == 0:
                    temp_coeffs[i] = cAn
                else:
                    temp_coeffs.append(None)
                    temp_coeffs[1] = cD[i - 1]

                reconstructed_signal.append(pywt.waverec(temp_coeffs, 'db1'))

            An = reconstructed_signal[0]
            D = reconstructed_signal[1:]

            for i in range(len(coeffs)):
                if i >= len(mav_data):
                    for j in range(3):
                        emg_feature_data[j].append([])

                mav_data[i].append(Utility.mean_absolute_value(coeffs[i]))
                rms_data[i].append(Utility.root_mean_square(coeffs[i]))
                wl_data[i].append(Utility.waveform_length(coeffs[i]))

        coeffs_size = len(mav_data)
        for i in range(3):
            for j in range(coeffs_size):
                emg_feature_data[i][j] = Utility.NormalizeArray(emg_feature_data[i][j])

        return numpy.array(emg_feature_data).flatten()

    def get_emg_data_features(self):
        return self.wavelet_feature_extraxtion()
