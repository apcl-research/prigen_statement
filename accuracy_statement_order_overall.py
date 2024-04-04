import pickle
import argparse
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='prigen_predictions/statement.txt')
    parser.add_argument('--prigen-filea', type=str, default='/nublar/datasets/prigen/prigen_statement/raw_data/testB.pkl')
    parser.add_argument('--prigen-fileb', type=str, default='/nublar/datasets/prigen/prigen_statement/raw_data/testA.pkl')

    args = parser.parse_args()
    input_file = args.input
    prigen_filea = args.prigen_filea
    prigen_fileb = args.prigen_fileb

    
    prigena = pickle.load(open(prigen_filea, 'rb'))
    prigenb = pickle.load(open(prigen_fileb, 'rb'))
    preds = dict()
    predicts = open(input_file, 'r')
    fidsb = []

    for res in prigenb:
        fid =  res['fid']
        fidsb.append(fid)


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
    count_zero_statement_matching = 0
    count_one_statement_matching = 0
    count_two_statement_matching = 0
    count_three_statement_matching = 0

    for resa in prigena:
        if(resa['fid'] not in preds):
            continue
        zero_matching = True 
        indexb = fidsb.index(resa['fid'])
        resb = prigenb[indexb]
        count_all += 1
        human_judgementa = [resa['first'], resa['second'], resa['third']]
        human_judgementb = [resb['first'], resb['second'], resb['third']]
        machine_prediction = preds[resa['fid']]
        if(human_judgementa[:] == machine_prediction[:] or human_judgementb[:] == machine_prediction[:]):
            count_three_statement_matching += 1
            zero_matching = False
        if(human_judgementa[:2] == machine_prediction[:2] or human_judgementb[:2] == machine_prediction[:2]):
            count_two_statement_matching += 1
            zero_matching = False
        if(human_judgementa[:1] == machine_prediction[:1] or human_judgementb[:1] == machine_prediction[:1]):
            count_one_statement_matching += 1
            zero_matching = False

        if(zero_matching):
            count_zero_statement_matching += 1

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
     
    #print(f'all samples:\t{count_all}')
    #print(f'number of samples that machine prediction matches human judgement:\t{count_one_statement_matching + count_two_statement_matching + count_three_statement_matching}')
    print(f'number of samples that are zero statement matching:\t{count_zero_statement_matching}')
    print(f'number of samples that are one statement matching:\t{count_one_statement_matching}')
    print(f'number of samples that are two statements matching:\t{count_two_statement_matching}')
    print(f'number of samples that are three statements matching:\t{count_three_statement_matching}')
    #print(f'accuracy:\t{accuracy}')
    
    
