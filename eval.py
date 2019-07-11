# coding: utf-8
# created by deng on 2019-03-22

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import End2EndDataset, gen_sentence_tensors
from utils.torch_util import calc_f1
from utils.path_util import from_project_root


def evaluate_e2e(model, data_url, bsl_model=None):
    """ evaluating end2end model on dataurl

    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model

    Returns:
        ret: dict of precision, recall, and f1

    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = End2EndDataset(data_url, next(model.parameters()).device, evaluating=True)
    loader = DataLoader(dataset, batch_size=200, collate_fn=dataset.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    sentence_true_list, sentence_pred_list = list(), list()
    region_true_list, region_pred_list = list(), list()
    region_true_count, region_pred_count = 0, 0

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        for data, sentence_labels, region_labels, records_list in loader:
            if bsl_model:
                pred_sentence_labels = torch.argmax(bsl_model.forward(*data), dim=1)
                pred_region_output, _ = model.forward(*data, pred_sentence_labels)
            else:
                try:
                    pred_region_output, pred_sentence_output = model.forward(*data)
                    # pred_sentence_output (batch_size, n_classes, lengths[0])
                    pred_sentence_labels = torch.argmax(pred_sentence_output, dim=1)
                    # pred_sentence_labels (batch_size, max_len)
                except RuntimeError:
                    print("all 0 tags, no evaluating this epoch")
                    return ret

            # pred_region_output (n_regions, n_tags)
            pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).cpu()
            # (n_regions)

            lengths = data[1]
            ind = 0
            for sent_labels, length, true_records in zip(pred_sentence_labels, lengths, records_list):
                pred_records = dict()
                for start in range(0, length):
                    if sent_labels[start] == 1:
                        for end in range(start + 1, length):
                            if sent_labels[end - 1] == 0:
                                break
                            if pred_region_labels[ind] > 0:
                                pred_records[(start, end)] = pred_region_labels[ind]
                            ind += 1

                region_true_count += len(true_records)
                region_pred_count += len(pred_records)

                for region in true_records:
                    true_label = dataset.label_list.index(true_records[region])
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

            pred_sentence_labels = pred_sentence_labels.view(-1).cpu()
            sentence_labels = sentence_labels.view(-1).cpu()
            for tv, pv, in zip(sentence_labels, pred_sentence_labels):
                sentence_true_list.append(tv)
                sentence_pred_list.append(pv)

        print("sentence binary labeling result:")
        print(classification_report(sentence_true_list, sentence_pred_list,
                                    target_names=['out-entity', 'in-entity'], digits=6))

        print("region classification result:")
        print(classification_report(region_true_list, region_pred_list,
                                    target_names=dataset.label_list, digits=6))
        ret = dict()
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)

    return ret


def predict(model, sentences, labels):
    """ predict NER result for sentence list
    Args:
        model: trained model
        sentences: sentences to be predicted
        labels: entity type list

    Returns:
        predicted results: ([sentence_labels], [region_labels], lengths)
    """

    tensors = gen_sentence_tensors(sentences, device=next(model.parameters()).device,
                                   data_url=from_project_root('data/genia/vocab.json'))
    pred_region_output, pred_sentence_output = model.forward(*tensors)
    pred_sentence_labels = torch.argmax(pred_sentence_output, dim=1).cpu()
    pred_region_labels = torch.argmax(pred_region_output, dim=1).cpu()

    lengths = tensors[1]
    pred_sentence_records = []

    ind = 0
    for sent_labels, length in zip(pred_sentence_labels, lengths):
        pred_records = dict()
        for start in range(0, length):
            if sent_labels[start] == 1:
                for end in range(start + 1, length):
                    if sent_labels[end - 1] == 0:
                        break
                    if pred_region_labels[ind] > 0:
                        pred_records[(start, end)] =  labels[pred_region_labels[ind]]
                    ind += 1
        pred_sentence_records.append(pred_records)

    return pred_sentence_labels, pred_sentence_records, lengths


def predict_on_iob2(model, iob_url):
    """ predict on iob2 file and save the results

    Args:
        model: trained model
        iob_url: url to iob file
    """

    save_url = iob_url.replace('.iob2', '.pred.txt')
    print("predicting on {} \n the result will be saved in {}".format(
        iob_url, save_url))
    test_set = End2EndDataset(iob_url, device=next(
        model.parameters()).device)

    model.eval()
    with open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
        for sentence, records in test_set:
            save_file.write(' '.join(sentence) + '\n')
            save_file.write("length = {} \n".format(len(sentence)))
            save_file.write("Gold records: {}\n".format(str(records)))
            try:
                sentence_labels, sentence_records, length =\
                    list(zip(*predict(model, [sentence], test_set.label_list)))[0]
            except RuntimeError as re:
                sentence_labels, sentence_records, length = "None", "None", len(sentence)
            save_file.write("Pred binary labels: {}\n".format(str(sentence_labels)))
            save_file.write("Pred records: {}\n".format(str(sentence_records)))
            if sentence_records != "None":
                details = str([sentence[rg[0]:rg[1]] for rg in sentence_records])
                save_file.write("{}\n\n".format(details))


def main():
    model_url = from_project_root("data/model/best_model.pt")
    print("loading model from", model_url)
    model = torch.load(model_url)
    # model = torch.load(model_url, map_location='cpu')
    test_url = from_project_root("data/genia/genia.test.iob2")
    evaluate_e2e(model, test_url)
    # predict_on_iob2(model, test_url)
    pass


if __name__ == '__main__':
    main()
