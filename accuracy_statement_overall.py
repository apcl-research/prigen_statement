import pickle
import argparse
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='prigen_predictions/statement.txt')
    parser.add_argument('--prigen-filea', type=str, default='/nublar/datasets/prigen/prigen_statement/testA.pkl')
    parser.add_argument('--prigen-fileb', type=str, default='/nublar/datasets/prigen/prigen_statement/testB.pkl')

    args = parser.parse_args()
    input_file = args.input
    prigen_filea = args.prigen_filea
    prigen_fileb = args.prigen_fileb

    
    prigena = pickle.load(open(prigen_filea, 'rb'))
    prigenb = pickle.load(open(prigen_fileb, 'rb'))
    


    preds = dict()
    predicts = open(input_file, 'r')

    allfidb = []

    for res in prigenb:
        fid = res['fid']
        allfidb.append(fid)

    for c, line in enumerate(predicts):
        pred = line.split('\t')
        preds[pred[0]] = [] 
        for i in range(2, len(pred)):
            if(i == len(pred) -1):
                pred[i] = pred[i].split('</s>')[0].strip()
            preds[pred[0]].append(pred[i].strip())
    predicts.close()
    count_acc = 0
    count_all = 0
    count_one_statement_matching = 0
    count_two_statement_matching = 0
    count_three_statement_matching = 0
    count_zero_statement_matching = 0
    label_functions = {}
    for indexa, resa in enumerate(prigena):
        if(resa['fid'] not in preds):
            continue

        labela = resa['label']
        codea = resa['code']
        indexb = allfidb.index(resa['fid'])
        resb = prigenb[indexb]
        if(labela not in label_functions):
            label_functions[labela] = []
        #if ( codea in label_functions[labela]):
        #    continue
        label_functions[labela].append(codea)

        count_all += 1
        human_judgementa = [resa['first'], resa['second'], resa['third']]
        human_judgementb = [resb['first'], resb['second'], resb['third']]
        machine_prediction = preds[resa['fid']]
        counta = 0
        index_lista = []
        countb = 0
        index_listb = []

        for index, stat in enumerate(human_judgementa):
            stat = stat.strip()
            for index_m, machine_pred in enumerate(machine_prediction):
                machine_pred = machine_pred.strip()
                if(stat == machine_pred and index_m not in index_lista):
                    index_lista.append(index_m)
                    counta += 1
                    break 
        for index, stat in enumerate(human_judgementb):
            stat = stat.strip()
            for index_m, machine_pred in enumerate(machine_prediction):
                machine_pred = machine_pred.strip()
                if(stat == machine_pred and index_m not in index_listb):
                    index_listb.append(index_m)
                    countb += 1
                    break 

        if(counta == 1 or countb == 1):
            count_one_statement_matching += 1
        if(counta == 2 or countb == 2):
            count_two_statement_matching += 1
        if(counta == 3 or countb ==3):
            count_three_statement_matching += 1
        if(counta == 0 and countb ==0):
            count_zero_statement_matching += 1 
    print(f'number of samples that are zero statement matching:\t{count_zero_statement_matching}')
    print(f'number of samples that are one statement matching:\t{count_one_statement_matching}')
    print(f'number of samples that are two statements matching:\t{count_two_statement_matching}')
    print(f'number of samples that are three statements matching:\t{count_three_statement_matching}')
    
    
