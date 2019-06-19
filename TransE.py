# TransE implemented by Afshin Sadeghi , in python 2.7
# you can run the train and the test separately by two functions, see the end of the file
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.matlib as matlib

sys.path.append('../')
import HandleKGDBs.ReadKGsDataset as ReadKGDataset

class SampleGenerator(nn.Module):
    def __init__(self, dataset_setting):
        super(SampleGenerator, self).__init__()  # Calling Super Class's constructor
        self.get_dataset(dataset_setting)

    def get_dataset(self, dataset_setting):
        sample_class = ReadKGDataset.readDataset()
        if dataset_setting.dataset == "FB15":
            sample_class.read_FB15K()
        elif dataset_setting.dataset == "WN18":
            sample_class.read_WN18()
        self.training_sample = torch.tensor(sample_class.train_data_)
        self.validation_samples = torch.tensor(sample_class.validation_data_)
        self.test_samples = torch.tensor(sample_class.test_data_)
        self.entity2id = sample_class.entity2id_
        self.relation2id = sample_class.relation2id_
        self.get_negative_samples = np.vectorize(self.get_negative_sample, signature='(),(),()->(n)')
        # print self.training_sample.shape[0]

    def get_random_training_samples(self, sample_size):
        training_index = np.random.randint(0, self.training_sample.shape[0], size=sample_size)
        training = self.training_sample[training_index]
        return training

    # add lables and makes one negative sample per positive sample
    def get_negative_sample(self, h, t, r):
        # randomly corrupt head and tails
        # X1 = [h, r, t]
        pos = np.random.randint(10, size=1)[0]
        pos2 = np.random.randint(self.entity2id.shape[0], size=1)[0]
        curr_entity = self.entity2id[pos2]
        #curr_entity = curr_entity[1] # now I only get the index , so only one column
        if (pos < 5):
            X2 = [h, curr_entity, r]
        else:
            X2 = [curr_entity, t, r]
        # x = np.array([X1, X2])
        # X2 = torch.tensor([X2])
        X2 = np.array(X2)
        # print X2
        # print
        return X2

    def reshuffle(self):
        #print "shuffleing.."
        training_index = torch.randperm(self.training_sample.shape[0])
        training = self.training_sample[training_index]
        self.training_sample = training

    # returns splitted training samples
    def get_splitted_set_training_batchs(self, sample_size):
        #print "get splitting set-..."
        return torch.split(self.training_sample, sample_size)
    #def get_splitted_set_negative_training_batchs(self, sample_size):
    #    return torch.split(self.negative_samples, sample_size)


class TransEModel(nn.Module):
    def __init__(self, config, sampler , embeddings):
        super(TransEModel, self).__init__()  # Calling Super Class's constructor
        self.config = config
        self.embeddings = embeddings
        self.sampler = sampler
        self.iter_ = 0
        self.x_drawing = np.zeros(1000)
        self.out_draw = np.zeros(1000)
        self.out_draw_negative_ = np.zeros(1000)

        self.batch_counter = 0
        #self.batch = self.sampler.get_splitted_set_training_batchs(self.config.x_train_bach_size)

    def make_splitted_batch(self):
        self.batch = self.sampler.get_splitted_set_training_batchs(self.config.x_train_bach_size)

    def get_splitted_variables(self):
        self.x_train = Variable(self.batch[self.batch_counter])
        self.x_train_negative = Variable(torch.tensor(
            self.sampler.get_negative_samples(self.x_train[:, 0], self.x_train[:, 1], self.x_train[:, 2])))

    def get_variables(self):
        self.x_train = Variable(self.sampler.get_random_training_samples(self.config.x_train_bach_size))
        # print x_train[0]
        # print x_train.shape
        self.x_train_negative = Variable(torch.tensor(
            self.sampler.get_negative_samples(self.x_train[:, 0], self.x_train[:, 1], self.x_train[:, 2])))
        # print x_train_negative.shape
        # clear grads as discussed in prev post
        # inputs_pos = Variable(h,r,t)
        # inputs_negative = Variable(h_negative,t_negative,r_negative)

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, False)  # .cuda()
        y = Variable(torch.Tensor([-1]))  # .cuda()
        loss = criterion(p_score, n_score, y)
        # lambda_var + score - score_negative, min_var
        return loss

    def forward(self):
        if self.config.batch_type == "random_batch":
            self.get_variables()
        elif self.config.batch_type == "pre_splitted_batch":
            #print self.batch_number
            if self.batch_counter == 0:
                self.sampler.reshuffle()
                self.make_splitted_batch()
                #print self.x_train[:, 0]
            self.batch_counter = self.batch_counter + 1
            self.get_splitted_variables()

        h = self.embeddings.get_vectorised_values_entity(self.x_train[:, 0])
        t = self.embeddings.get_vectorised_values_entity(self.x_train[:, 1])
        r = self.embeddings.get_vectorised_values_relation(self.x_train[:, 2])
        h_negative = self.embeddings.get_vectorised_values_entity(self.x_train_negative[:, 0])
        t_negative = self.embeddings.get_vectorised_values_entity(self.x_train_negative[:, 1])
        r_negative = self.embeddings.get_vectorised_values_relation(self.x_train_negative[:, 2])
        # x = F.relu(self.linear(x))
        #print "h"
        #print h.shape
        #print "r"
        #print r.shape  # X  dimension is [Nx, m]or  [Nx, training_example_number]  where Nx is feature numbers in x and m is the sample number
        if self.config.L1_Norm:
            score_pos = torch.norm((h + r - t), p=1, dim=1)
            score_neg = torch.norm((h_negative + r_negative - t_negative), p=1, dim=1)
        else:
            score_pos = torch.norm((h + r - t), p=2, dim=1)
            score_neg = torch.norm((h_negative + r_negative - t_negative), p=2, dim=1)
        #score = self.model.forward(h, r, t)
        #score_negative = self.model.forward(h_negative, r_negative, t_negative)
        # print outputs.shape
        loss = self.loss_func(score_pos, score_neg)
        if self.iter_ < self.config.x_train_bach_size - 1 & self.iter_ < 100:
            self.x_drawing[self.iter_] =  self.iter_
            self.out_draw[self.iter_] = score_pos.item()  # score[batch_counter].data.numpy()
            self.out_draw_negative_[self.iter_] = score_neg.item()
            self.iter_ = self.iter_ + 1

        elif self.iter_ == 100:
            # iter_ = 0
            plt.plot(self.x_drawing, self.out_draw, 'go', label='positive', alpha=.5)
            plt.plot(self.x_drawing, self.out_draw_negative_, 'r.', label='negative', alpha=0.5)
            plt.legend()
            plt.draw()
            plt.pause(0.1)
            plt.clf()
            self.iter_ = 0
        return loss

    def predict(self, h,t,r):
        if self.config.L1_Norm:
            score= torch.norm((h + r - t), p=1, dim=1)
        else:
            score = torch.norm((h + r - t), p=2, dim=1)
        return score


class Embeddings(nn.Module):
    def __init__(self, sampler, config):
        super(Embeddings, self).__init__()  # Calling Super Class's constructor
        self.entity_embedding = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.relation_embedding = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)

    # for vector of elements
    def get_vectorised_values_entity(self, x):
        a = self.entity_embedding(torch.LongTensor(x))
        return a

    def get_vectorised_values_relation(self, x):
        a = self.relation_embedding(torch.LongTensor(x))
        return a

    def get_vectorised_value_relation(self, x):
        a = self.relation_embedding(x)
        return a

    def get_vectorised_value_entity(self, x):
        a = self.entity_embedding(x)
        return a


class Experiment(object):
    def __init__(self, dataset_setting):
        self.dataset_setting = dataset_setting
        self.sampler = SampleGenerator(self.dataset_setting)
        self.config = HyperParameters(self.dataset_setting, self.sampler)
        self.embeddings = Embeddings(self.sampler, self.config)
        self.model = TransEModel(self.config,self.sampler, self.embeddings)  # .double()

        self.mean_rank = 0
        self.hit_ten_tail = 0
        self.hit_one_tail = 0
        self.hit_three_tail = 0
        self.hit_ten_head = 0
        self.hit_one_head = 0
        self.hit_three_head = 0
        self.hit_hundred_tail = 0
        self.hit_hundred_head = 0

    def train(self):

        # y_correct = np.zeros(50)
        lambda_ = torch.FloatTensor(np.random.uniform(-1 / 5, 1 / 5, [self.config.x_train_bach_size]))
        zero_vector = torch.FloatTensor(np.random.uniform(-000000.1, 000000.1, [self.config.x_train_bach_size]))
        min_var = Variable(zero_vector)
        lambda_var = Variable(lambda_)

        for epoch in range(0, self.config.epochs):
            sum_loss = 0
            if self.config.batch_type == "pre_splitted_batch":
                #self.sampler.reshuffle()
                #self.model.get_splitted_variables()
                self.model.batch_counter = 0

            for batch_counter in range(0, self.config.number_of_batch):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
                #optimizer = torch.optim.Adagrad(self.model.parameters(),lr=self.config.learning_rate)
                optimizer.zero_grad()
                loss =self.model()
                loss.backward()  # back props
                optimizer.step()  # update the parameters
                self.embeddings.entity_embedding.weight.data.renorm_(p=2, dim=0, maxnorm=1)
                self.embeddings.relation_embedding.weight.data.renorm_(p=2, dim=0, maxnorm=1)
                sum_loss = sum_loss + loss.item()
            loss_per_triple = (sum_loss) / (self.sampler.training_sample.shape[0])
            print('epoch {}, loss_per_triple {}'.format(epoch, loss_per_triple))
            if epoch > 0 and epoch % 50 == 0:
                self.save(epoch)
                #self.test()
        print ("training finished with epochs, learning rate, dataset" + str(self.config.epochs) + " "+ str(self.config.learning_rate) + " "+ str(self.config.x_feature_dimension)+ " "+self.dataset_setting.dataset)

    def save(self, epoch):
        torch.save(self.model.embeddings, self.config.result_dir +"/transE" + str(epoch)+ self.dataset_setting.dataset+ str(self.config.x_feature_dimension))
        return

    def load_db_embeddings(self,dataset_setting):
        self.model.embeddings = Embeddings(self.sampler, self.config)
        path1 = self.config.result_dir + "/transE" + str(self.config.epochs) + dataset_setting.dataset + str(
                self.config.x_feature_dimension)
        self.model.embeddings = torch.load(path1)
        return self.model.embeddings

    def test(self):
        print ("testing on "+ self.dataset_setting.dataset)
        for triple in self.sampler.test_samples:  # testing with vaidation set .test_samples:
            # print triple
            R = self.model.embeddings.get_vectorised_value_relation(triple[2]).unsqueeze(0)

            score_test = self.model.predict(self.model.embeddings.get_vectorised_value_entity(triple[0]),
                                            self.model.embeddings.get_vectorised_value_entity(triple[
                                                                                            1]), R).detach().numpy()[0]
            reproduce_head = matlib.repmat(triple, self.model.sampler.entity2id.shape[0], 1)
            reproduce_head[:, 0] = self.model.sampler.entity2id#[:, 1]
            reproduce_tail = matlib.repmat(triple, self.model.sampler.entity2id.shape[0], 1)
            reproduce_tail[:, 1] = self.model.sampler.entity2id#[:, 1]
            score_test_head = self.model.predict(self.model.embeddings.get_vectorised_values_entity(reproduce_head[:, 0]),
                                                 self.model.embeddings.get_vectorised_values_entity(reproduce_head[:, 1]),
                                                 self.model.embeddings.get_vectorised_values_relation(
                                                     reproduce_head[:, 2])).detach().numpy()
            score_test_tail = self.model.predict(self.model.embeddings.get_vectorised_values_entity(reproduce_tail[:, 0]),
                                                 self.model.embeddings.get_vectorised_values_entity(reproduce_tail[:, 1]),
                                                 self.model.embeddings.get_vectorised_values_relation(
                                                     reproduce_tail[:, 2])).detach().numpy()
            try:
                score_test_head = np.sort(score_test_head, None)
                score_test_tail = np.sort(score_test_tail, None)
                hit_head = np.amin(np.where(score_test_head == score_test)[0])
                hit_tail = np.amin(np.where(score_test_tail == score_test)[0])
                #print hit_head
                #print hit_tail
                if hit_tail < 101:
                    self.hit_hundred_tail = self.hit_hundred_tail + 1
                if hit_head < 101:
                    self.hit_hundred_head = self.hit_hundred_head + 1
                if hit_tail < 11:
                    self.hit_ten_tail = self.hit_ten_tail + 1
                if hit_head < 11:
                    self.hit_ten_head = self.hit_ten_head + 1
                if hit_tail < 2:
                    self.hit_one_tail = self.hit_one_tail + 1
                if hit_head < 2:
                    self.hit_one_head = self.hit_one_head + 1
                if hit_tail < 4:
                    self.hit_three_tail = self.hit_three_tail + 1
                if hit_head < 4:
                    self.hit_three_head = self.hit_three_head + 1

                self.mean_rank = self.mean_rank + hit_head + hit_tail
            except ValueError:  # raised if `score_test_head` is empty.
                pass
        print ("hit at 1,3,10, 100 and mean rank:")
        print ((self.hit_one_tail + self.hit_one_head) / (self.sampler.test_samples.shape[0] * 2))
        print ((self.hit_three_tail + self.hit_three_head) / (self.sampler.test_samples.shape[0] * 2))
        print ((self.hit_ten_tail + self.hit_ten_head) / (self.sampler.test_samples.shape[0] * 2))
        print ((self.hit_hundred_tail + self.hit_hundred_head) / (self.sampler.test_samples.shape[0] * 2))
        print (self.mean_rank / (self.sampler.test_samples.shape[0] * 2))
        self.mean_rank = 0
        self.hit_ten_tail = 0
        self.hit_one_tail = 0
        self.hit_three_tail = 0
        self.hit_ten_head = 0
        self.hit_one_head = 0
        self.hit_three_head = 0
        self.hit_hundred_tail = 0
        self.hit_hundred_head = 0
        return


class DatasetSetting(object):
    def __init__(self):
        self.dataset =  "dbp15k-zh"# "dbp15k-ja" "dbp15k-en"  # "stexpan" "membeta" #FB15 WN18 dbp15k-en dbp15k-fr
    def set_dataset(self, dataset_name):
        self.dataset = dataset_name

class HyperParameters(object):
    def __init__(self, dataset_setting, sampler):
        self.dataset_setting = dataset_setting
        self.x_feature_dimension = 75
        self.epochs = 1500
        #self.x_feature_dimension = 50
        self.L1_Norm = False
        self.margin = 1.0
        if self.dataset_setting.dataset == "membeta":
            self.x_feature_dimension = 100 # the number triples is about size times to fb15k so 50. with 20 result become worse , with 100 was bad also
            self.margin = 1.0
            self.L1_Norm = False #L2
            self.epochs = 2000

        elif self.dataset_setting.dataset == "stexpan":
            self.x_feature_dimension = 100 # 20 gives better result, using 50 to make them comparible  # 20 the number triples is less than the WN18. now testing 50 because number of relations are match more
            self.margin = 1.0
            self.L1_Norm = False
            self.epochs = 2000

        elif self.dataset_setting.dataset == "FB15":
            self.x_feature_dimension = 50
            self.margin = 1.0
            self.L1_Norm = False #L2
        elif self.dataset_setting.dataset == "WN18":
            self.x_feature_dimension = 20  # 20 for wn, for fb 50
            self.margin = 1.0
            self.L1_Norm = True
        self.r_feature_dimension = self.x_feature_dimension
        self.entity = 0
        self.relation = 0
        self.learning_rate = 0.01
        self.number_of_batch = 100
        self.x_train_bach_size = int(sampler.training_sample.shape[0] / self.number_of_batch)
        self.result_dir = "tmp"
        self.batch_type = "pre_splitted_batch"  #pre_splitted_batch  random_batch
        # relation_number = 8  # for wn

def run_experiment():
    experiment_dataset_setting = DatasetSetting()
    experiment = Experiment(experiment_dataset_setting)
    experiment.train()
    experiment.save(experiment.config.epochs) # so that to store the model last epoch as well.
    experiment.test()

def load_model_test_experiment():
    experiment_dataset_setting = DatasetSetting()
    experiment = Experiment(experiment_dataset_setting)
    experiment.load_db_embeddings(experiment_dataset_setting)
    experiment.test()

def load_model_():
    print ("testing if model for datasets can be loaded")
    experiment_dataset_setting = DatasetSetting()
    experiment = Experiment(experiment_dataset_setting)
    experiment.load_db_embeddings(experiment_dataset_setting)
    print ("model for dataset" + experiment_dataset_setting.dataset + " is loaded")
    return


#run_experiment()
#load_model_test_experiment()
#load_model_

