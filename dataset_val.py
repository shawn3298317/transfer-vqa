import os
import operator
import numpy as np

from six.moves import cPickle as pickle
from collections import defaultdict
from external.vqa.vqa import VQA

image_dir = "/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/val2014"
img_prefix = "COCO_val2014_"
qjson = "/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json"
ajson = "/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/Annotations_Val_mscoco/mscoco_val2014_annotations.json"

with open('./data/a2i.pkl', 'rb') as f:
    a2i = pickle.load(f)

vqa = VQA(ajson, qjson)

img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
img_ids = []
for fname in img_names:
    img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
    img_ids.append(int(img_id))

ques_ids = vqa.getQuesIds(img_ids)

ques_ids_modif = []
for ques_id in ques_ids:
    qa = vqa.loadQA(ques_id)[0]
    qqa = vqa.loadQQA(ques_id)[0]

    ques = qqa['question'][:-1]
    answers = qa['answers']
    answer = ""
    for ans in answers:
        ans = ans['answer'].lower()
        if ans in a2i:
            answer = ans
            break

    if not answer == "":
        ques_ids_modif.append(ques_id)

np.save('./data/va_img_names.npy', img_names)
np.save('./data/va_img_ids.npy', img_ids)
np.save('./data/va_ques_ids.npy', ques_ids_modif)
