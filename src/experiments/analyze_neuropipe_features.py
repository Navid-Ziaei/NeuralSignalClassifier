import os
import pickle

directory_path = "D://XDFv3//"
file_list = os.listdir(directory_path)
file_path = directory_path + file_list[0]

with open(file_path, 'rb') as file:
    features = pickle.load(file)