import pickle
import argparse
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='prigen_predictions/statement.txt')
    parser.add_argument('--prigen-file', type=str, default='/nublar/datasets/prigen/prigen_statement/raw_data/testA.pkl')

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

    for res in prigen:
        if(res['fid'] not in preds):
            continue
        count_all += 1
        human_judgement = [res['first'], res['second'], res['third']]
        machine_prediction = preds[res['fid']]
        if(human_judgement[:] == machine_prediction[:]):
            count_three_statement_matching += 1
        elif(human_judgement[:2] == machine_prediction[:2]):
            count_two_statement_matching += 1
        elif(human_judgement[:1] == machine_prediction[:1]):
            count_one_statement_matching += 1

        #count_temp = 0
        #for stat in human_judgement:
        #    if (stat in machine_prediction):
        #        count_temp += 1
        
        #if(count_temp == 1):
        #    count_one_statement_matching += 1
        #elif(count_temp == 2):
        #    count_two_statement_matching += 1
        #elif(count_temp == 3):
        #    count_three_statement_matching += 1
     
    accuracy = (count_one_statement_matching + count_two_statement_matching + count_three_statement_matching) / count_all
    #print(f'all samples:\t{count_all}')
    #print(f'number of samples that machine prediction matches human judgement:\t{count_one_statement_matching + count_two_statement_matching + count_three_statement_matching}')
    print(f'number of samples that are one statement matching:\t{count_one_statement_matching}')
    print(f'number of samples that are two statements matching:\t{count_two_statement_matching}')
    print(f'number of samples that are three statements matching:\t{count_three_statement_matching}')
    #print(f'accuracy:\t{accuracy}')
    
    
