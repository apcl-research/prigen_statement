import pickle
import tiktoken

testfids = pickle.load(open("/nublar/datasets/prigen/prigen_statement/purpose_advertisement/testfid_advertisement.pkl", 'rb'))
valfids = pickle.load(open("/nublar/datasets/prigen/prigen_statement/purpose_advertisement/valfid_advertisement.pkl", 'rb'))



statements = pickle.load(open("/nublar/datasets/prigen/prigen_statement/purpose_advertisement/unique_filter.pkl", "rb"))
enc = tiktoken.get_encoding("gpt2")

total_len = 0
total_samples = 0
for statement in statements:
    fid = statement['fid']
    if(fid in testfids or fid in valfids):
        continue
    code = statement['code']
    first_hop = "FIRST HOP:\t" + statement['code'] + "\n"
    label = statement["label"]
    statement1 = statement['first']
    statement2 = statement['second']
    statement3 = statement['third']
    prompt = first_hop + f"LABEL:\t{label}\nSTATEMENT:<s>\t{statement1}\t{statement2}\t{statement3}</s>"
    prompt_len = len(enc.encode(prompt))
    total_len += prompt_len
    total_samples += 1

print(total_len / total_samples)
print(total_samples)

