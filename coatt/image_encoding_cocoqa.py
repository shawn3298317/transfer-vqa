import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from six.moves import cPickle as pickle
from tqdm import tqdm

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CocoQAImgDataset(Dataset):

    def __init__(self,
                 image_dir,
                 image_names,
                 img_prefix,
                 name):

        """
        Args:
            image_dir (string): Path to the directory with COCO images
            image_names_file (string): Path to the npy file containing image name array
            name (string): Train or Test

        """
        self.image_dir = image_dir
        self.image_names = image_names
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        img_ids = {}
        for idx, fname in enumerate(self.image_names):
            img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
            img_ids[int(img_id)] = idx

        with open('./data/cocoqa/' + name + '_enc_idx_res50.npy', 'wb') as f:
            pickle.dump(img_ids, f)


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = default_loader(self.image_dir + '/' + self.image_names[idx])
        imgT = self.transform(img).float()

        return imgT

tr_img_names = np.load('./data/cocoqa/tr_img_names.npy', encoding='latin1').tolist()
va_img_names = np.load('./data/cocoqa/va_img_names.npy', encoding='latin1').tolist()

tr_image_dir = '/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/train2014/'
va_image_dir = '/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/val2014/'
tr_out_dir = './data/cocoqa/tr_enc_res50'
va_out_dir = './data/cocoqa/va_enc_res50'
cnn_type = "resnet50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Init pre-trained CNN
if cnn_type == "resnet18":
    model = models.resnet18(pretrained=True)
elif cnn_type == "resnet50":
    model = models.resnet50(pretrained=True)
elif cnn_type == "vgg":
    model = models.vgg16(pretrained=True)

modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
for params in model.parameters():
    params.requires_grad = False
if DEVICE == 'cuda':
    model = model.cuda()
model.eval()


tr_img_dataset = CocoQAImgDataset(tr_image_dir, tr_img_names, "COCO_train2014_", "train")
tr_img_dataset_loader = DataLoader(tr_img_dataset, batch_size=1, shuffle=False, num_workers=2)

va_img_dataset = CocoQAImgDataset(va_image_dir, va_img_names, "COCO_val2014_", "val")
va_img_dataset_loader = DataLoader(va_img_dataset, batch_size=1, shuffle=False, num_workers=2)

print('Dumping Training images encodings.')
for idx, imgT in enumerate(tqdm(tr_img_dataset_loader)):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = tr_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    #print(path)
    #print(out.shape)

print('Dumping Validation images encodings.')
for idx, imgT in enumerate(tqdm(va_img_dataset_loader)):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = va_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)

