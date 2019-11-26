import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from six.moves import cPickle as pickle
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from coatt.dataset import pre_process_dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CocoQADataset(Dataset):

    def __init__(self,
                 image_dir,
                 question_id_file,
                 image_names,
                 answer_text_file,
                 collate=False,
                 q2i=None,
                 a2i=None,
                 i2a=None,
                 method="simple"):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_id_file (string): Path to the npy file containing the question id array
            image_names_file (string): Path to the npy file containing image name array
            answer_text_file (string): Path to the text file containing answertext

        """

        print(method)
        self.image_dir = image_dir
        self.ques_ids = np.load(question_id_file, allow_pickle=True)
        #self.image_names = np.load(image_names_file, allow_pickle=True)
        self.image_names = image_names
        self.answers = [s.strip() for s in open(answer_text_file, "r").readlines()]
        self.q2i = q2i
        self.a2i = a2i
        self.i2a = i2a
        self.method = method
        self.collate = collate

        self.transform = transforms.Compose([transforms.Resize((448, 448)),
                                             transforms.ToTensor()])

        self.q2i_len = len(self.q2i)
        self.a2i_len = len(self.a2i.keys())
        self.q2i_keys = self.q2i.keys()


    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, idx):
        img = default_loader(self.image_dir + '/' + self.image_names[idx])
        #imgT = self.transform(img).permute(1, 2, 0)
        imgT = self.transform(img).float()

        ques_id = self.ques_ids[idx]
        quesT = torch.from_numpy(np.array(ques_id)).long()
        answer = self.answers[idx]

        if answer == "":
            gT = torch.from_numpy(np.array([len(self.a2i)])).long()
        else:
            gT = torch.from_numpy(np.array([self.a2i[answer]])).long()

        if not self.collate:
            return {'img' : imgT, 'ques' : quesT, 'gt': gT}

        return imgT, quesT, gT
