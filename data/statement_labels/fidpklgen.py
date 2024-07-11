import pickle
import random


valsize = 100

all_dat = pickle.load(open('/nublar/datasets/prigen/prigen_statement/purpose_advertisement/prigen_advertisement_all.pkl', 'rb'))
allfids = list(all_dat.keys())
test_fids = pickle.load(open('/nublar/datasets/prigen/prigentestfids.pkl', 'rb'))
print(test_fids.keys())
test_fids = test_fids['purpose_advertisement']

#print(len(allfids))
#print(len(test_fids))

train_fids = list(set(allfids) - set(test_fids))

random.shuffle(train_fids)


#trainset = sample_id_list[:len(sample_id_list)-testsize-valsize]
trainset = train_fids[0:len(train_fids)-valsize]
valset = train_fids[len(train_fids)-valsize:]


#testset = sample_id_list[len(sample_id_list)-testsize:]

#for i in trainset:
#    print(dat[i]['label'])


with open('/nublar/datasets/prigen/prigen_statement/purpose_advertisement/trainfid_advertisement.pkl', 'wb') as file:
    pickle.dump(trainset, file)
with open('/nublar/datasets/prigen/prigen_statement/purpose_advertisement/valfid_advertisement.pkl', 'wb') as file:
    pickle.dump(valset, file)
with open('/nublar/datasets/prigen/prigen_statement/purpose_advertisement/testfid_advertisement.pkl', 'wb') as file1:
    pickle.dump(test_fids, file1)


