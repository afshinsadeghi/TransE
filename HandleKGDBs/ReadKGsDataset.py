import numpy as np
import os

# I downloaded these datasets are from converted them to nt and then to id files.
# The download address of the rdf/xml files is http://oaei.ontologymatching.org/2018/knowledgegraph/index.html
class readDataset():

    def __init__(self):
        home_dir = os.getenv("HOME")
        self.input_directory = home_dir + "/Dropbox/ArticleArchive/EmbedDBs/" # /EmbedDBs/ on the server
        return

    def read_WN18(self, dataset_dir_=""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "WN18" 
        else:
            self.dataset_dir = dataset_dir_
        entity2id_file = self.dataset_dir + "/entity2id.txt"
        relation2id_file = self.dataset_dir + "/relation2id.txt"
        train_data_file = self.dataset_dir + "/train2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        vaidation_data_file = self.dataset_dir + "/valid2id.txt"
        self.entity2id_ = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t",usecols=(1), skiprows=1, dtype="int")
        self.relation2id_ = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")
        self.train_data_ = np.loadtxt(open(train_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )
        self.test_data_ = np.loadtxt(open(test_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )
        self.validation_data_ = np.loadtxt(open(vaidation_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )

    def read_FB15K(self, dataset_dir_ = ""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "FB15K"  # "FB15K" is the FB15K made by bordes
        else:
            self.dataset_dir = dataset_dir_
        entity2id_file =  self.dataset_dir + "/entity2id.txt"
        relation2id_file =  self.dataset_dir + "/relation2id.txt"
        train_data_file = self.dataset_dir + "/triple2id.txt"
        validation_data= self.dataset_dir + "/valid2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        self.entity2id_ = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")
        self.relation2id_ = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")
        self.entity_and_id_ = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(0,1),skiprows=1, dtype="str")
        self.relation_and_id_ = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(0,1),skiprows=1, dtype="str")

        self.train_data_ = np.loadtxt(open(train_data_file, "rb"), delimiter="\t", skiprows=1, dtype="int")
        self.validation_data_ = np.loadtxt(open(validation_data, "rb"), delimiter="\t", skiprows=1, dtype="int")
        self.test_data_ = np.loadtxt(open(test_data_file, "rb"), delimiter="\t", skiprows=1, dtype="int")