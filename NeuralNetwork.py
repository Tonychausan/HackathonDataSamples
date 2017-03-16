import numpy
import os
import time
import tensorflow as tf

import DataHandler
from DataHandler import DataHandler

import Utility
from Utility import File


TRAINING_DATA_FILE_PATH = 'NeuralNetwork/training_file.data'

SESS_PATH = 'NeuralNetwork/Sessions/{}/'.format("2017-03-16-1558")
SESS_MODEL_PATH = SESS_PATH + 'emg_model'

LEARNING_RATE = 0.05
N_EPOCH = 5000

layer_sizes = [0, 150, 8*3, 0]  # Network build


def create_emg_training_file():
    file_list = []
    folder = Utility.FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue

        file = File(filename, folder)

        if file.example_id <= 1500:
            file_list.append(file)

    data_handler = DataHandler(file_list[0])

    size_of_training_set = len(file_list)
    n_input_nodes = len(data_handler.get_emg_data_features())
    n_output_nodes = Utility.NUMBER_OF_GESTURES

    print("Creating EMG-training file...")
    with open(TRAINING_DATA_FILE_PATH, 'w') as outfile:
        print("training size:", size_of_training_set)
        outfile.write(str(size_of_training_set) + " ")

        print("Number of input neurons:", n_input_nodes)
        outfile.write(str(n_input_nodes) + " ")

        print("Number of output neurons:", n_output_nodes)
        outfile.write(str(n_output_nodes) + "\n")

        print()

        for data_file in file_list:
            print(data_file.filename)
            data_handler = DataHandler(data_file)

            emg_sums = data_handler.get_emg_data_features()
            for i in range(n_input_nodes):
                outfile.write(str(emg_sums[i]))
                if i < n_input_nodes - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")

            for gesture in range(n_output_nodes):
                if gesture != data_file.gesture:
                    outfile.write("0")
                else:
                    outfile.write("1")

                if gesture < Utility.NUMBER_OF_GESTURES - 1:
                    outfile.write(" ")
                else:
                    outfile.write("\n")


def create_network_meta_data_file(path):
    file_path = path + "network.meta"
    with open(file_path, 'w') as outfile:
        outfile.write("layer_sizes: ")
        for layer_size in layer_sizes:
            outfile.write(str(layer_size) + " ")
        outfile.write("\n")

        outfile.write("Epoch_count 0")


def get_network_meta_data_from_file():
    file_path = SESS_PATH + "network.meta"
    with open(file_path, 'r') as metafile:
        layer_size_list = metafile.readline().split()[1:]
        epoch_count = int(metafile.readline().split(" ")[1])

    return (list(map(int, layer_size_list)), epoch_count)


def update_epoch_count_meta_data(epoch_count):
    file_path = SESS_PATH + "network.meta"

    with open(file_path, 'r') as metafile:
        lines = metafile.readlines()

    lines[1] = "Epoch_count " + str(epoch_count)
    with open(file_path, 'w') as metafile:
        for line in lines:
            metafile.write(line)


def create_emg_network_variables(number_of_neuron_for_layer):
    number_of_variables = len(number_of_neuron_for_layer) - 1
    return_variables = []
    bias_variables = []

    for i in range(number_of_variables):
        variable_name = "theta" + str(i)
        variable = tf.Variable(tf.random_uniform([number_of_neuron_for_layer[i], number_of_neuron_for_layer[i + 1]], -1, 1), name=variable_name)
        return_variables.append(variable)

        bias_name = "bias" + str(i)
        bias = tf.Variable(tf.zeros(number_of_neuron_for_layer[i + 1]), name=bias_name)
        bias_variables.append(bias)

    return (return_variables, bias_variables)


def create_emg_network_layers(input_placeholder, variables, bias_variables):
    layers = []
    current_layer = input_placeholder
    for theta, bias in zip(variables, bias_variables):
        layer = tf.sigmoid(tf.matmul(current_layer, theta) + bias)
        layers.append(layer)
        current_layer = layer

    output = layers.pop()

    return (layers, output)


def get_training_inputs_and_outputs():
    inputs = []
    outputs = []

    with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

        line_counter = 0
        for line in training_data_file:
            if line_counter % 2 == 0:
                inputs.append([float(x) for x in line.split()])
            else:
                outputs.append([float(x) for x in line.split()])

            line_counter += 1

    return (inputs, outputs)


def get_training_meta_data():
    with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
        (training_size, n_inputs, n_outputs) = training_data_file.readline().split()

    return(int(training_size), int(n_inputs), int(n_outputs))


def create_emg_network():
    sess_path = 'NeuralNetwork/Sessions/{}/'.format(time.strftime("%Y-%m-%d-%H%M"))
    if os.path.exists(sess_path):
        run_or_not = input("A session with this name already exist, replace it? (y/n): ")
        if not run_or_not == "y":
            return

    print("Creating EMG-network")
    (inputs, outputs) = get_training_inputs_and_outputs()

    (training_size, n_inputs, n_outputs) = get_training_meta_data()

    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")

    layer_sizes[0] = n_inputs
    layer_sizes[-1] = n_outputs

    (theta, bias) = create_emg_network_variables(layer_sizes)
    (layers, output) = create_emg_network_layers(input_placeholder, theta, bias)

    cost = tf.reduce_mean(tf.square(outputs - output))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    if not os.path.exists(sess_path):
        os.makedirs(sess_path)

    sess_model_path = sess_path + 'emg_model'
    saver.save(sess, sess_model_path)
    create_network_meta_data_file(sess_path)  # Write meta data of session to file

    print("EMG-network created")
    print("Session path:", sess_model_path)
    tf.reset_default_graph()


def train_emg_network():
    print("Train Network")
    print("Training file:", TRAINING_DATA_FILE_PATH)
    print("Training session:", SESS_PATH)

    (inputs, outputs) = get_training_inputs_and_outputs()
    (training_size, n_inputs, n_outputs) = get_training_meta_data()
    (sess_layer_sizes, old_epoch_count) = get_network_meta_data_from_file()

    if(n_inputs != sess_layer_sizes[0] or n_outputs != sess_layer_sizes[-1]):
        print("Training file and session is not compatible!")
        return

    dummy = False
    while not dummy:
        n_steps = input("Number of steps: ")
        dummy = Utility.check_int_input(n_steps)
        n_steps = int(n_steps)

    dummy = False
    while not dummy:
        run_time = input("Max Time (hours): ")
        dummy = Utility.check_int_input(run_time)

    run_time = float(run_time) * 3600

    start_time = time.time()
    current_time = time.time() - start_time

    i = 0
    number_of_saved = 0
    while current_time < run_time and i < n_steps:
        continue_emg_network_training(sess_layer_sizes, inputs, outputs, n_inputs, n_outputs, training_size, n_steps, i)

        os.system('cls')
        print("Training Network")
        print("Training file:", TRAINING_DATA_FILE_PATH)
        print("Training session:", SESS_PATH)
        print("Number of steps:", n_steps)
        print("Max Time (hours):", run_time / 3600)
        print()

        number_of_saved += 1
        print("Number of saved:", number_of_saved)

        if i + N_EPOCH <= n_steps:
            i += N_EPOCH
        else:
            i += (n_steps % N_EPOCH)

        current_time = time.time() - start_time
        (hours, minutes, seconds) = Utility.second_to_HMS(current_time)
        print('Current time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

        if i == 0:
            estimated_time = 0
        else:
            estimated_time = (current_time / i) * (n_steps)
        (hours, minutes, seconds) = Utility.second_to_HMS(estimated_time)
        print('Estimated time: {:.0f}h {:.0f}min {:.0f}sec'.format(hours, minutes, seconds))

        print('Batch:', i)
        update_epoch_count_meta_data(old_epoch_count + i)

    print()
    print("Runtime:", '{0:.2f}'.format(float(time.time() - start_time)) + "sec")
    print("finished")


def continue_emg_network_training(sess_layer_sizes, inputs, outputs, n_inputs, n_outputs, training_size, n_steps, epoch_count):
    input_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_inputs], name="input")
    output_placeholder = tf.placeholder(tf.float32, shape=[training_size, n_outputs], name="output")

    (theta, bias) = create_emg_network_variables(sess_layer_sizes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SESS_MODEL_PATH)

        (layer, output) = create_emg_network_layers(input_placeholder, theta, bias)

        cost = tf.reduce_mean(tf.square(outputs - output))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        for i in range(N_EPOCH):
            if epoch_count + i >= n_steps:
                break
            sess.run(train_step, feed_dict={input_placeholder: inputs, output_placeholder: outputs})

        saver.save(sess, SESS_MODEL_PATH)

    tf.reset_default_graph()


def test_emg_network():
    file_list = []
    summary_list = []

    folder = Utility.FOLDER_NAME
    for filename in os.listdir(folder):
        if filename == ".gitignore":
            continue
        file = File(filename, folder)


        if file.example_id > 1500:
            file_list.append(file)

    for test_file in file_list:
        data_handler = DataHandler(test_file)

        start_time = time.time()
        results = input_test_emg_network(data_handler)
        end_time = time.time()

        recognized_gesture = numpy.argmax(results)
        print_results(results)

        print("Correct gesture:", test_file.gesture)
        print("Analyse time: ", "%.2f" % float(end_time - start_time))

        summary_list.append((test_file.gesture, recognized_gesture))

        print()
        print("File:", test_file.filename)

    print("#############################################################")
    print("Summary List")

    success_list = []
    for i in range(Utility.NUMBER_OF_GESTURES):
        success_list.append([0, 0])

    for correct_gesture, recognized_gesture in summary_list:

        success_list[correct_gesture][0] += 1

        if correct_gesture == recognized_gesture:
            success_list[correct_gesture][1] += 1

        print(correct_gesture, " -> ", recognized_gesture)

    print()
    print("#############################################################")
    print("Success Rate")
    for i in range(Utility.NUMBER_OF_GESTURES):
        print('{:d}\t{:4d} of {:4d}'.format(i, success_list[i][1], success_list[i][0]))


def input_test_emg_network(input_data_handler):
    test_inputs = [input_data_handler.get_emg_data_features()]

    (sess_layer_sizes, epoch_count) = get_network_meta_data_from_file()
    input_placeholder = tf.placeholder(tf.float32, shape=[1, sess_layer_sizes[0]], name="input")

    (theta, bias) = create_emg_network_variables(sess_layer_sizes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SESS_MODEL_PATH)

        (layers, output) = create_emg_network_layers(input_placeholder, theta, bias)

        results = sess.run(output, feed_dict={input_placeholder: test_inputs})

    tf.reset_default_graph()
    return results


def print_results(results):
    for result in results:
        print("\n###########################################################")
        for gesture in range(Utility.NUMBER_OF_GESTURES):
            print('{:d}\t{:10f}'.format(gesture, result[gesture]))

    print()
    print("Recognized: " + str(numpy.argmax(results)))


# create_emg_training_file()
# create_emg_network()
train_emg_network()
# test_emg_network()
