# coding: utf-8
# created by deng on 2019-01-23

from utils.torch_util import set_random_seed

RANDOM_SEED = 233
set_random_seed(RANDOM_SEED)

import os
import sys
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

import utils.json_util as ju
from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import End2EndDataset, prepare_vocab
from model import End2EndModel
from eval import evaluate_e2e

MAX_REGION = 10
EARLY_STOP = 5
LR = 0.0005
BATCH_SIZE = 50
MAX_GRAD_NORM = 5
FREEZE_WV = False
LOG_PER_BATCH = 20

# EMBD_URL = None  # not use pre_trained embeddings
EMBD_URL = from_project_root("data/embedding/PubMed-shuffle-win-30.bin")
VOCAB_URL = from_project_root("data/genia/vocab.json")
TRAIN_URL = from_project_root("data/genia/genia.train.iob2")
DEV_URL = from_project_root("data/genia/genia.dev.iob2")
TEST_URL = from_project_root("data/genia/genia.test.iob2")


def train_end2end(n_epochs=30,
                  embedding_url=None,
                  char_feat_dim=100,
                  freeze=FREEZE_WV,
                  train_url=TRAIN_URL,
                  dev_url=DEV_URL,
                  test_url=None,
                  learning_rate=LR,
                  batch_size=BATCH_SIZE,
                  early_stop=EARLY_STOP,
                  clip_norm=MAX_GRAD_NORM,
                  bsl_model_url=None,
                  gamma=0.6,
                  device='auto',
                  save_only_best=True
                  ):
    """ Train deep exhaustive model, trained best model will be saved at 'data/model/'

    Args:
        n_epochs: number of epochs
        embedding_url: url to pre-trained embedding file, set as None to use random embedding
        char_feat_dim: size of character level feature
        freeze: whether to freeze embedding
        train_url: url to train data
        dev_url: url to dev data
        test_url: urt to test data
        learning_rate: learning rate
        batch_size: batch_size
        early_stop: early stop for training
        clip_norm: whether to perform norm clipping, set to 0 if not need
        bsl_model_url: pre-trained sequence labeler url
        gamma: percentage of region classification module loss in total loss
        device: device for torch
        save_only_best: only save model of best performance
    """

    # print arguments
    arguments = json.dumps(vars(), indent=2)
    print("arguments", arguments)
    start_time = datetime.now()

    device = get_device(device)
    train_set = End2EndDataset(train_url, device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)

    vocab = ju.load(VOCAB_URL)
    n_words = len(vocab)
    char_vocab = ju.load(VOCAB_URL.replace('vocab', 'char_vocab'))
    n_chars = len(char_vocab)

    model = End2EndModel(
        hidden_size=200,
        lstm_layers=1,
        n_tags=train_set.n_tags,
        char_feat_dim=char_feat_dim,
        embedding_url=embedding_url,
        bidirectional=True,
        n_embeddings=n_words,
        embedding_dim=200,
        n_chars=n_chars,
        freeze=freeze
    )

    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
        # model = nn.DataParallel(model)
    else:
        print("using cpu\n")
    model = model.to(device)
    bsl_model = torch.load(bsl_model_url) if bsl_model_url else None

    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None
    for epoch in range(n_epochs):
        # switch to train mode
        model.train()
        batch_id = 0
        for data, sentence_labels, region_labels in train_loader:
            optimizer.zero_grad()
            pred_region_labels, pred_sentence_labels = model.forward(*data, sentence_labels)
            classification_loss = criterion(pred_region_labels, region_labels)
            bsl_loss = criterion(pred_sentence_labels, sentence_labels)
            if bsl_model_url:
                # train condition region classifier alone
                loss = classification_loss
            else:
                # train region classifier and binary sequence labeler as a multitask learning
                loss = gamma * classification_loss + (1 - gamma) * bsl_loss
            loss.backward()

            # gradient clipping
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            endl = '\n' if batch_id % LOG_PER_BATCH == 0 else '\r'
            sys.stdout.write("epoch #%d, batch #%d, loss: %.6f, %s%s" %
                             (epoch, batch_id, loss.item(), datetime.now().strftime("%X"), endl))
            sys.stdout.flush()
            batch_id += 1

        print('\n')

        cnt += 1
        # evaluating model use development dataset or and additional test dataset
        precision, recall, f1 = evaluate_e2e(model, dev_url, bsl_model).values()
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            name = 'split' if bsl_model else 'end2end'
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = from_project_root(
                "data/model/%s_model_epoch%d_%f.pt" % (name, epoch, f1))
            torch.save(model, best_model_url)
            cnt = 0

        # if test_url:
        #     evaluate_e2e(model, test_url, bsl_model)

        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))

        if cnt >= early_stop > 0:
            break

    if test_url:
        best_model = torch.load(best_model_url)
        print("best model url:", best_model_url)
        print("evaluating on test dataset:", test_url)
        evaluate_e2e(best_model, test_url, bsl_model)

    print(arguments)


def main():
    start_time = datetime.now()
    pretrained_url = prepare_vocab([TRAIN_URL, DEV_URL, TEST_URL],
                                   EMBD_URL, update=False, min_count=1)
    train_end2end(test_url=TEST_URL, embedding_url=pretrained_url)
    print("finished in:", datetime.now() - start_time)
    pass


if __name__ == '__main__':
    main()
