import pickle
import argparse
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='prigen_predictions/statement.txt')
    parser.add_argument('--prigen-file', type=str, default='/nublar/datasets/prigen/prigen_statement/testA.pkl')

    args = parser.parse_args()
    input_file = args.input
    prigen_file = args.prigen_file

    
    prigen = pickle.load(open(prigen_file, 'rb'))
    
    preds = dict()
    predicts = open(input_file, 'r')


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
    label_functions = {}
    for res in prigen:
        if(res['fid'] not in preds):
            continue
        label = res['label']
        code = res['code']
        if(label not in label_functions):
            label_functions[label] = []
        if ( code in label_functions[label]):
            continue
        label_functions[label].append(code)

        count_all += 1
        human_judgement = [res['first'], res['second'], res['third']]
        machine_prediction = preds[res['fid']]
        count_temp = 0
        index_list = []
        for index, stat in enumerate(human_judgement):
            stat = stat.strip()
            for index_m, machine_pred in enumerate(machine_prediction):
                machine_pred = machine_pred.strip()
                if(stat == machine_pred and index_m not in index_list):
                    index_list.append(index_m)
                    count_temp += 1
                    break 
        if(count_temp == 1):
            count_one_statement_matching += 1
        elif(count_temp == 2):
            count_two_statement_matching += 1
        elif(count_temp == 3):
            count_three_statement_matching += 1
     
    accuracy = (count_one_statement_matching + count_two_statement_matching + count_three_statement_matching) / count_all
    print(f'all samples:\t{count_all}')
    print(f'number of samples that machine prediction matches human judgement:\t{count_one_statement_matching + count_two_statement_matching + count_three_statement_matching}')
    print(f'number of samples that are one statement matching:\t{count_one_statement_matching}')
    print(f'number of samples that are two statements matching:\t{count_two_statement_matching}')
    print(f'number of samples that are three statements matching:\t{count_three_statement_matching}')
    print(f'accuracy:\t{accuracy}')
    
    
