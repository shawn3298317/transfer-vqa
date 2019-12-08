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

class CocoQAXDataset(Dataset):
    def __init__(self,
                 image_enc_dir,
                 image_names,
                 question_id_file,
                 answer_text_file,
                 img_prefix,
                 collate=False,
                 enc_idx=None,
                 q2i=None,
                 a2i=None,
                 i2a=None):

        self.image_enc_dir = image_enc_dir
        self.image_names = image_names
        self.ques_ids = np.load(question_id_file, allow_pickle=True)
        self.answers = [s.strip() for s in open(answer_text_file, "r").readlines()]
        self.enc_idx = enc_idx
        self.q2i = q2i
        self.a2i = a2i
        self.i2a = i2a
        self.collate = collate

        self.q2i_len = len(self.q2i)
        self.a2i_len = len(self.a2i.keys())
        self.q2i_keys = self.q2i.keys()

        self.img_ids = []
        for fname in self.image_names:
            img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
            self.img_ids.append(int(img_id))

    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        file_idx = self.enc_idx[img_id]
        path = self.image_enc_dir + '/' + str(file_idx) + '.npz'
        img = np.load(path)['out']               # 512 x 196
        imgT = torch.from_numpy(img).float()

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
