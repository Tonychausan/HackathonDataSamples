import numpy
import pywt

import Utility


EMG_WAVELET_LEVEL = 1
NUMBER_OF_FEATURES = 3

feature_functions = [Utility.mean_absolute_value, Utility.root_mean_square, Utility.waveform_length]


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
        emg_feature_data = []

        n = EMG_WAVELET_LEVEL
        for emg_id in range(len(self.emg_data)):
            emg_array = self.emg_data[emg_id]

            coefficient_subsets = pywt.wavedec(emg_array, 'db1', level=n)
            cAn = coefficient_subsets[0]  # approximation coefficient
            cD = coefficient_subsets[1:]  # detail coefficient subset

            reconstructed_signals = []
            for i in range(n + 1):
                temp_coeffs = [None] * (n + 1 - i)  # init list [cAn, cD(n-i), cD(n-i-1)..., cD1]
                if i == 0:
                    temp_coeffs[0] = cAn
                else:
                    temp_coeffs.append(None)
                    temp_coeffs[1] = cD[i - 1]

                reconstructed_signals.append(pywt.waverec(temp_coeffs, 'db1'))

            # Emg signal = An + Dn + D(n-1) + ... + D1
            # An = reconstructed_signals[0]
            # D = reconstructed_signals[1:]

            number_of_coeffcient_subsets = len(coefficient_subsets)
            number_of_reconstructed_signals = len(reconstructed_signals)

            for feature_function_id in range(len(feature_functions)):
                feature_function = feature_functions[feature_function_id]

                if emg_id == 0:
                    for i in range(number_of_coeffcient_subsets):
                        emg_feature_data.append([feature_function(coefficient_subsets[i])])

                    for i in range(number_of_reconstructed_signals):
                        emg_feature_data.append([feature_function(reconstructed_signals[i])])
                else:
                    emg_feature_data_id = feature_function_id * (number_of_coeffcient_subsets + number_of_reconstructed_signals)
                    for i in range(number_of_coeffcient_subsets):
                        j = emg_feature_data_id + i
                        emg_feature_data[j].append(feature_function(coefficient_subsets[i]))

                    for i in range(number_of_reconstructed_signals):
                        j = emg_feature_data_id + number_of_coeffcient_subsets + i
                        emg_feature_data[j].append(feature_function(reconstructed_signals[i]))

        for i in range(len(emg_feature_data)):
            emg_feature_data[i] = Utility.NormalizeArray(emg_feature_data[i])

        return numpy.array(emg_feature_data).flatten()

    def raw_emg_feature_extraxtion(self):
        emg_feature_data = []
        for emg_id in range(len(self.emg_data)):
            emg_array = self.emg_data[emg_id]
            for feature_function_id in range(len(feature_functions)):
                feature_function = feature_functions[feature_function_id]
                if emg_id == 0:
                    emg_feature_data.append([feature_function(emg_array)])
                else:
                    emg_feature_data[feature_function_id].append(feature_function(emg_array))

        for i in range(len(emg_feature_data)):
            emg_feature_data[i] = Utility.NormalizeArray(emg_feature_data[i])

        return emg_feature_data

    def get_emg_data_features(self):
        emg_data_features = numpy.append(self.wavelet_feature_extraxtion(), self.raw_emg_feature_extraxtion())
        return emg_data_features
