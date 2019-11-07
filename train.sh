#!/bin/bash

DATASET_PATH="/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014/"
python main.py --model coattention\
               --train_image_dir ${DATASET_PATH}train2014/\
               --train_question_path ${DATASET_PATH}Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json\
               --train_annotation_path ${DATASET_PATH}Annotations_Train_mscoco/mscoco_train2014_annotations.json\
               --test_image_dir ${DATASET_PATH}val2014/\
               --test_question_path ${DATASET_PATH}Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json\
               --test_annotation_path ${DATASET_PATH}Annotations_Val_mscoco/mscoco_val2014_annotations.json\
               --batch_size 256\
               --num_epochs 100\
               --num_data_loader_workers 4
