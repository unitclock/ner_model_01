import json
from tqdm import tqdm
import pandas as pd

unique_labels = [
    "dis",
    "sym",
    "pro",
    "equ",
    "dru",
    "ite",
    "bod",
    "dep",
    "mic",
]

bio_unique_labels = []
bio_unique_labels.extend([ "B-"+l for l in unique_labels])
bio_unique_labels.extend([ "I-"+l for l in unique_labels])
bio_unique_labels.extend([ "O-"+l for l in unique_labels])
bio_unique_labels.extend([ "E-"+l for l in unique_labels])
bio_unique_labels.extend([ "S-"+l for l in unique_labels])
bio_unique_labels.append("O")

labels_to_ids = {k: v for v, k in enumerate(sorted(bio_unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(bio_unique_labels))}

def handle_raw_data(raw_path:str,csv_path:str):
    pre_pd = pd.DataFrame(columns=["W","B"])
    with open(raw_path,"r") as file:
        raw_data = json.load(file)
        raw_data_bar = tqdm(raw_data)
        for r in raw_data_bar:
            words = r["text"]
            if len(r["text"]) > 510:
                words = words[0:511]
            bio = ["O"]*len(r["text"])
            for e in r["entities"]:
                e_type = e["type"]
                e_start =int(e["start_idx"])
                e_end = int(e["end_idx"])
                if e_end >511:
                    continue
                if e_end - e_start == 1 :
                    bio[e_start] = "S-"+e_type
                elif e_end- e_start == 2:
                    bio[e_start] = "B-"+e_type
                    bio[e_end-1]="E-"+e_type
                elif e_end - e_start >= 3:
                    bio[e_start] = "B-"+e_type
                    for i in range(e_start+1,e_end-1):
                        bio[i]="I-"+e_type
                    bio[e_end-1]="E-"+e_type
                else:
                    print("error")
            words = list(words)
            if len(words) != len(bio) or len(words)>510 or len(bio)>510:
                print(len(words),len(bio))
                
            pre_pd.loc[len(pre_pd)] = {"W":words,"B":bio}
    pre_pd.to_csv(csv_path)
    return pre_pd
                    