import torch
import numpy as np
from six.moves import cPickle as pickle
from torch.utils.data import DataLoader

from coatt.coattention_net import CoattentionNet
from coatt.experiment_runner_base import ExperimentRunnerBase
from coatt.cocoqa_dataset import CocoQADataset


def collate_lines(seq_list):
    imgT, quesT, gT = zip(*seq_list)
    lens = [len(ques) for ques in quesT]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    imgT = torch.stack([imgT[i] for i in seq_order])
    quesT = [quesT[i] for i in seq_order]
    gT = torch.stack([gT[i] for i in seq_order])
    return imgT, quesT, gT


class CoattentionNetCocoExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self,
                 train_image_dir,
                 train_question_path,
                 train_annotation_path,
                 test_image_dir,
                 test_question_path,
                 test_annotation_path,
                 batch_size,
                 num_epochs,
                 num_data_loader_workers,
                 lr):

        self.method = 'coattention'
        print('Loading numpy files. \n')
        with open('./data/cocoqa/q2i.pkl', 'rb') as f:
            q2i = pickle.load(f)
        with open('./data/cocoqa/a2i.pkl', 'rb') as f:
            a2i = pickle.load(f)
        with open('./data/cocoqa/i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)

        tr_img_names = np.load('./data/cocoqa/tr_img_names.npy', encoding='latin1').tolist()
        #tr_img_ids = np.load('./data/cocoqa/tr_img_ids.npy', encoding='latin1').tolist()
        #tr_ques_ids = np.load('./data/cocoqa/tr_ques_ids.npy', encoding='latin1').tolist()

        va_img_names = np.load('./data/cocoqa/va_img_names.npy', encoding='latin1').tolist()
        #va_img_ids = np.load('./data/cocoqa/va_img_ids.npy', encoding='latin1').tolist()
        #va_ques_ids = np.load('./data/cocoqa/va_ques_ids.npy', encoding='latin1').tolist()

        print('Creating Datasets.')
        train_dataset = CocoQADataset(image_dir=train_image_dir,
                                      question_id_file=train_question_path,
                                      answer_text_file=train_annotation_path,
                                      collate=True,
                                      image_names=tr_img_names,
                                      q2i=q2i,
                                      a2i=a2i,
                                      i2a=i2a,
                                      method=self.method)

        val_dataset = CocoQADataset(image_dir=test_image_dir,
                                    question_id_file=test_question_path,
                                    answer_text_file=test_annotation_path,
                                    collate=True,
                                    image_names=va_img_names,
                                    q2i=q2i,
                                    a2i=a2i,
                                    i2a=i2a,
                                    method=self.method)

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers, collate_fn=collate_lines)

        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers, collate_fn=collate_lines)


        print('Creating Co Attention Model.')
        model = CoattentionNet(len(q2i), 512).float()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)


    def _optimize(self, predicted_answers, true_answer_ids):
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_answers, true_answer_ids)
        loss.backward()
        self.optimizer.step()

        return loss
