import numpy
import os
import re

NUMBER_OF_GESTURES = 2
FOLDER_NAME = 'GestureData/'


class File:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        (self.gesture, self.example_id) = self.get_data_from_filename(filename)

    def get_file_path(self):
        return self.path + self.filename

    def get_data_from_filename(self, filename):
        pattern = r'Gesture([1-6])_Example([0-9]+).CSV'
        matchObj = re.match(pattern, filename, re.I)

        return (int(matchObj.group(1)) - 1, int(matchObj.group(2)))


def transpose_data_files():
    folder = FOLDER_NAME
    for filename in os.listdir(folder):
        infilename = os.path.join(folder, filename)
        print(infilename)

        my_data = numpy.genfromtxt(infilename, delimiter=',')
        my_data = numpy.array(my_data)
        my_data = my_data.transpose()

        numpy.savetxt(infilename, my_data, fmt='%i', delimiter=",")


def NormalizeArray(array):
    return array / numpy.linalg.norm(array)


def mean_absolute_value(values):
    absolute_values = numpy.absolute(values)
    return numpy.mean(absolute_values)


def root_mean_square(values):
    square_value = numpy.square(values)
    N = square_value.size
    sum_value = numpy.sum(square_value)
    return numpy.sqrt((1 / N) * sum_value)


def waveform_length(values):
    diff_values = numpy.subtract(values[:len(values) - 1], values[1:])
    absolute__diff_values = numpy.absolute(diff_values)
    sum_absolute_diff_values = numpy.sum(absolute__diff_values)
    return sum_absolute_diff_values


def second_to_HMS(current_time):
    hours = current_time // 3600
    current_time %= 3600
    minutes = current_time // 60
    current_time %= 60
    seconds = current_time

    return (hours, minutes, seconds)


def check_int_input(i):
    try:
        i = float(i)
    except ValueError:
        print("That's not an int!")
        return False

    return True
