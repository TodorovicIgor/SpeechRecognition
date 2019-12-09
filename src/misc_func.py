from scipy.io import wavfile
import os
import csv
import random
import numpy as np
import tensorflow.keras as k
import matplotlib.pyplot as plt


labels = {
    "zero": 0,
    "one": 1,    
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "on": 10,
    "off": 11,
    "no": 12,
    "yes": 13,
    "stop": 14,
    "go": 15,
    "left": 16,
    "right": 17,
    "up": 18,
    "down": 19,

    "bed": 20,
    "bird": 20,
    "cat": 20,
    "dog": 20,
    "happy": 20,
    "house": 20,
    "marvin": 20,
    "sheila": 20,
    "tree": 20,
    "wow": 20,
}
inverted_labels = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "on",
    11: "off",
    12: "no",
    13: "yes",
    14: "stop",
    15: "go",
    16: "left",
    17: "right",
    18: "up",
    19: "down",
    20: "unkown"
}


def generate_labels():
    train_audio = os.listdir("../data/train/audio")
    with open('../data/train/training.csv', mode='w+') as csv_file:
        for folder in train_audio:
            if folder != "_background_noise_":
                files = os.listdir("../data/train/audio/"+folder)
                for file in files:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    csv_writer.writerow(["../data/train/audio/"+folder+"/"+file, labels[folder]])
    # shuffling data
    with open('../data/train/training.csv', mode='r') as csv_file:
        lines = csv_file.readlines()
        random.shuffle(lines)
        training_lines = lines[0:int(len(lines)*0.8)]
        testing_lines = lines[int(len(lines)*0.8):len(lines)]

    # writing to csv shuffled data for training
    with open('../data/train/training.csv', mode='w+') as csv_file:
        csv_file.writelines(training_lines)

    # writing to csv shuffled data for testing
    with open('../data/train/testing.csv', mode='w+') as csv_file:
        csv_file.writelines(testing_lines)


def read_data_from_csv(batch_size=0):
    # returns [[16000],label]
    ret_data, ret_label = [], []
    data, label = [], []
    rows = []
    if batch_size == 0:
        with open('../data/train/training.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for row in csv_reader:
                row_data = read_wav(row[0])
                for _ in range(len(row_data)+1, 16001):
                    row_data.append(0)
                np_row = np.array(row_data)
                rows.append(np_row)
                label.append(int(row[1]))
        data = np.array(rows)
        label = np.array(label)
        return data, label
    else:
        num_lines = sumlines('../data/train/training.csv')
        num_batches = int(num_lines/batch_size)
        with open('../data/train/training.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            # row_count = sum(1 for row in csv_reader)
            # print("row_count is", row_count)
            # print("batch_size is", batch_size)

            row_counter = 1
            batch_counter = 0
            for row in csv_reader:
                row_counter += 1
                row_data = read_wav(row[0])
                for _ in range(len(row_data)+1, 16001):
                    row_data.append(0)
                np_row = np.array(row_data)
                rows.append(np_row)
                label.append(int(row[1]))
                print(num_batches, batch_counter)

                if row_counter == batch_size:
                    row_counter = 1
                    batch_counter += 1

                    if batch_counter == num_batches:
                        return ret_data, ret_label
                    data = np.array(rows)
                    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))  # data.shape[0]=2000, data.shape[1]=16000
                    ret_data.append(data)
                    ret_label.append(np.array(label))
        return "UNEXPECTED", "UNEXPECTED"


def read_batch(batch_size=0, batch_num=0):
    data, label = [], []
    ret_data, ret_label = [], []
    rows = []
    if batch_size > 0:
        with open('../data/train/training.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            lines = csv_file.readlines()
            lines = lines[batch_num*batch_size:(batch_num+1)*batch_size]
            # print("len of lines should be", batch_size, "and is", len(lines))
            for row in lines:
                row_data = read_wav(row.split(',')[0])
                for _ in range(len(row_data) + 1, 16001):
                    row_data.append(0)
                np_row = np.array(row_data)
                rows.append(np_row)
                label.append(int(row.split(',')[1]))
        data = np.array(rows)
        label = np.array(label)
    return data, label


def prepare_dataset(data, labels):
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))  # data.shape[0]=2000, data.shape[1]=16000
    labels = k.utils.to_categorical(labels)
    return data, labels



def prepare_dataset3(data, labels):
    """
    splitting 16000 features to 4*4000
    2000 samples goes to 8000
    """
    data = np.reshape(data, (data.shape[0]*4, data.shape[1]//4, 1))  # data.shape[0]=2000, data.shape[1]=16000
    # print(data[0])
    labels = k.utils.to_categorical(labels)
    labels = [val for val in labels for _ in range(4)]
    labels = np.array(labels)
    labels.astype(int)
    return data, labels


def sumlines(filename):
    with open(filename) as f:
        return sum(1 for line in f)


def read_single_data(path):
    data = []
    row_data = read_wav(path)
    for _ in range(len(row_data) + 1, 16001):
        row_data.append(0)
    np_row = np.array(row_data)
    data.append(np_row)
    data = np.reshape(data, (1, 1, 16000))
    return data


def read_wav(path):
    br, data = wavfile.read(path)
    # print(data)

    max_nb_bit = float(2 ** (16 - 1))
    normalized = data / (max_nb_bit + 1.0)
    # plt.plot(normalized.tolist())
    # plt.show()
    return normalized.tolist()


if __name__ == "__main__":
    read_wav("../data/train/audio/dog/0ab3b47d_nohash_0.wav")
    # generate_labels()
    # read_data_from_csv()
    # print(generate_raw_input(10))
