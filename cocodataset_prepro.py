import os
import pickle
import itertools
import numpy as np
from collections import Counter

from tqdm import tqdm
from nltk.tokenize import word_tokenize


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(questions, params):

    # preprocess all the question
    tokens = []
    for s in tqdm(questions):
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)
        tokens.append(txt)

    return tokens

def build_vocab_question(questions, params):
    # build vocabulary for question and answers.

    # count up the number of words
    counts = Counter(itertools.chain(*questions))
    threshold = min(params['word_count_threshold'], len(counts)) if "word_count_threshold" in params else len(counts)
    counts_thr = counts.most_common(threshold)
    print(counts_thr[:20])
    cw = {tup[0]: i+3 for i, tup in enumerate(counts_thr)}

    # print some stats
    total_words = sum(counts.values())
    print("Total words:", total_words)
    #bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    print("inserting the special UNK token")
    cw["UNK"] = 0
    cw["<sos>"] = 1
    cw["<eos>"] = 2
    vocab = ["UNK", "<sos>", "<eos>"] + [tup[0] for tup in counts_thr]
    #bad_count = sum(counts[w] for w in bad_words)
    #print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print("number of words in vocab = ", len(vocab))
    #print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

    question_ids = []
    tot_len = 0.0
    for q in questions:
        to_add = [cw["<sos>"]]

        for word in q:
            to_add.append(cw.get(word, cw["UNK"]))
        to_add += [cw["<eos>"]]
        question_ids.append(to_add)
        tot_len += len(to_add)

    print("Average train question length:", tot_len/len(questions))
    return question_ids, cw, vocab

def encode_question(questions, q_cw):
    question_ids = []
    tot_len = 0.0
    for q in questions:
        to_add = [q_cw["<sos>"]]

        for word in q:
            to_add.append(q_cw.get(word, q_cw["UNK"]))
        to_add += [q_cw["<eos>"]]
        question_ids.append(to_add)
        tot_len += len(to_add)

    print("Average test question length:", tot_len/len(questions))
    return question_ids


def get_top_answers(answers, params):

    counts = Counter(answers).most_common()
    cw = {tup[0]: i for i, tup in enumerate(counts)}
    vocab = [tup[0] for tup in counts]
    print("Top answers:", len(vocab), vocab[:10])

    return cw, vocab


if __name__ == "__main__":

    param = {
        "token_method": "nltk"
    }

    COCO_PATH = "/projectnb/statnlp/shawnlin/dataset/cocoqa_2015/"
    VQA_IMG_PATH = "/projectnb/statnlp/shawnlin/dataset/mscoco_vqa_2014"

    with open(COCO_PATH + "train/questions.txt", "r") as f:
        raw_question_sent_tr = [s.strip() for s in f.readlines()]
    tr_ques = prepro_question(raw_question_sent_tr, param)

    train_ques_ids, q_cw, q_vocab = build_vocab_question(tr_ques, param)
    print(train_ques_ids[0])

    with open(COCO_PATH + "test/questions.txt", "r") as f:
        raw_question_sent_tst = [s.strip() for s in f.readlines()]
    test_ques = prepro_question(raw_question_sent_tst, param)
    test_ques_ids = encode_question(test_ques, q_cw)
    print(test_ques_ids[0])

    with open(COCO_PATH + "train/answers.txt", "r") as f:
        raw_answer = [s.strip() for s in f.readlines()[:]]
    a_cw, a_vocab = get_top_answers(raw_answer, param)

    img_dir = VQA_IMG_PATH + "/train2014/"
    img_prefix = "COCO_train2014_"
    with open(COCO_PATH + "train/img_ids.txt", "r") as f:
        train_img_names = ["%s%012d.jpg" % (img_prefix, int(s.strip())) for s in f.readlines()]

    img_dir = VQA_IMG_PATH + "/val2014/"
    img_prefix = "COCO_val2014_"
    with open(COCO_PATH + "test/img_ids.txt", "r") as f:
        test_img_names = ["%s%012d.jpg" % (img_prefix, int(s.strip())) for s in f.readlines()]

    with open("./data/cocoqa/q2i.pkl", "wb") as f_q2i:
        pickle.dump(q_cw, f_q2i)
    with open("./data/cocoqa/a2i.pkl", "wb") as f_a2i:
        pickle.dump(a_cw, f_a2i)
    with open("./data/cocoqa/i2a.pkl", "wb") as f_i2a:
        pickle.dump(a_vocab, f_i2a)

    np.save("./data/cocoqa/tr_ques_ids.npy", np.array(train_ques_ids))
    np.save("./data/cocoqa/va_ques_ids.npy", np.array(test_ques_ids))
    np.save("./data/cocoqa/tr_img_names.npy", train_img_names)
    np.save("./data/cocoqa/va_img_names.npy", test_img_names)
