import os
import json
import numpy as np
import random


def traverse_directory(path):
    directory_structure = {}

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            directory_structure[item] = traverse_directory(item_path)
        else:
            directory_structure[item] = "file"

    return directory_structure


def split_train_val_data(dir_structure_json, train_ratio=0.9):

    all_imgs = []
    def _recursive_find(item, path):
        if isinstance(item, dict):
            for key, value in item.items():
                if value == "file":
                    all_imgs.append(path + "/" + key)
                _recursive_find(value, path + "/" + key)
        elif isinstance(item, list):
            for i,element in enumerate(item):
                _recursive_find(element,f"{i}")
        else:
            if item != "file":
                all_imgs.append(path + "/" + item)

    _recursive_find(dir_structure_json, "")
    
    all_imgs = list(set(all_imgs))
    unique_imgs = list(set([item.split("/")[-1] for item in all_imgs]))

    cut_point = int(train_ratio * len(unique_imgs))
    unique_train_imgs, unique_val_imgs = unique_imgs[:cut_point], unique_imgs[cut_point:]   
    print(f"Number of train, val unique images: {len(unique_train_imgs)}, {len(unique_val_imgs)}")
    
    train_imgs = [item for item in all_imgs if item.split("/")[-1] in unique_train_imgs]
    val_imgs = [item for item in all_imgs if item.split("/")[-1] in unique_val_imgs]
    print(f"Number of train, val real images: {len(train_imgs)}, {len(val_imgs)}")

    with open(splits_train_path, 'w') as f:
        json.dump(train_imgs, f, indent=4)
    with open(splits_val_path, 'w') as f:
        json.dump(val_imgs, f, indent=4)
    
    return unique_train_imgs, unique_val_imgs


def generate_data(structure, imgs_group, num_per_class, scale_factor=[1,1,1]):

    output = []

    D1 = structure['d1']
    D2 = structure['d2']
    D3 = structure['d3']
    
    MAX_RETRY = 1000

    def pattern1(D, scale):
        # type: similar (A)
        sum_all = 0
        for type_, imgs_ in D.items():
            #print("type_: ", type_)
            history_pairs = []
            sum_per = 0
            imgs_ = list(imgs_.keys())
            retry_times = 0
            s = type_.count("+") + 1
            while sum_per < num_per_class * scale and \
                    retry_times < MAX_RETRY:
                #print("sum_per: ", sum_per)
                pair = random.sample(imgs_, 2)
                if pair[0] not in imgs_group or pair[1] not in imgs_group:
                    continue
                if pair not in history_pairs:
                    history_pairs.append(pair)
                    d = {
                        "candidate": f"/d{s}/" + type_ + "/" + pair[0],
                        "target": f"/d{s}/" + type_ + "/" + pair[1],
                        #"captions": [f"similar degradation {type_}"]
                        "captions": [f"similar degradation"]
                        }
                    output.append(d)
                    sum_per += 1
                    sum_all += 1
                    retry_times = 0
                else:
                    retry_times += 1
        return output, sum_all

    def pattern2(D1, D2, scale):
        # type:  (A, A+B), (A+B, A)
        sum_all_0, sum_all_1 = 0, 0
        for type_0, imgs_0 in D1.items():
            imgs_0 = list(imgs_0.keys())
            len_0 = type_0.count("+") + 1
            for type_1, imgs_1 in D2.items():
                imgs_1 = list(imgs_1.keys())
                len_1 = type_1.count("+") + 1
                
                set_0, set_1 = set(type_0.split("+")), set(type_1.split("+"))
                if not (set_0.issuperset(set_1) or set_1.issuperset(set_0)):
                    continue
                
                diff_ = ",".join(set_1-set_0) if set_1.issuperset(set_0) else ",".join(set_0-set_1)

                sum_0 = 0
                retry_times = 0
                history_pairs = []
                while sum_0 < num_per_class * scale \
                        and retry_times < MAX_RETRY:
                    pair = [random.choice(imgs_0), random.choice(imgs_1)]
                    if pair[0] not in imgs_group or pair[1] not in imgs_group:
                        continue
                    if pair not in history_pairs:
                        history_pairs.append(pair)
                        d = {
                            "candidate": f"/d{len_0}/" + type_0 + "/" + pair[0],
                            "target": f"/d{len_1}/" + type_1 + "/" + pair[1],
                            "captions": [f"less degradation {diff_}"]
                            }
                        output.append(d)
                        sum_0 += 1
                        sum_all_0 += 1
                        retry_times = 0
                    else:
                        retry_times += 1

                sum_1 = 0
                retry_times = 0
                history_pairs = []
                while sum_1 < num_per_class * scale \
                        and retry_times < MAX_RETRY:
                    pair = [random.choice(imgs_0), random.choice(imgs_1)]
                    if pair[0] not in imgs_group or pair[1] not in imgs_group:
                        continue
                    if pair not in history_pairs:
                        history_pairs.append(pair)
                        d = {
                            "candidate": f"/d{len_1}/" + type_1 + "/" + pair[1],
                            "target": f"/d{len_0}/" +type_0 + "/" + pair[0],
                            "captions": [f"more degradation {diff_}"]
                            }
                        output.append(d)
                        sum_1 += 1
                        sum_all_1 += 1
                        retry_times = 0
                    else:
                        retry_times += 1
        return output, sum_all_0, sum_all_1
    
    output, sum_d1 = pattern1(D1, scale_factor[0]*6)
    print("Number of data for pattern (A, A): ", sum_d1)
    output, sum_d1 = pattern1(D2, scale_factor[0]*2)
    print("Number of data for pattern (A+B, A+B): ", sum_d1)
    output, sum_d1 = pattern1(D3, scale_factor[0])
    print("Number of data for pattern (A+B+C, A+B+C): ", sum_d1)
    output, sum_d1_d2, sum_d2_d1 = pattern2(D1, D2, scale_factor[1]*4)
    print("Number of data for pattern (A, A+B): ", sum_d1_d2)
    print("Number of data for pattern (A+B, A): ", sum_d2_d1)
    output, sum_d1_d3, sum_d3_d1 = pattern2(D1, D3, scale_factor[1]*2)
    print("Number of data for pattern (A, A+B+C): ", sum_d1_d3)
    print("Number of data for pattern (A+B+C, A): ", sum_d3_d1)
    output, sum_d2_d3, sum_d3_d2 = pattern2(D2, D3, scale_factor[2])
    print("Number of data for pattern (A+B, A+B+C): ", sum_d2_d3)
    print("Number of data for pattern (A+B+C, A+B): ", sum_d3_d2)
    
    return output


# root dir
root_dir = "/home/jason/CLIP4Cir/ImgRestore_dataset"

# input
raw_image_dir = os.path.join(root_dir, "images/LQ")

# output
dir_structure_json_path = os.path.join(root_dir, "structure/dir_structure.json")

cap_train_path = os.path.join(root_dir, "captions/cap.train.json")
cap_val_path = os.path.join(root_dir, "captions/cap.val.json")
splits_train_path = os.path.join(root_dir, "image_splits/split.train.json")
splits_val_path = os.path.join(root_dir, "image_splits/split.val.json")

# params
train_val_ratio = 0.9

# get structure
structure = traverse_directory(raw_image_dir)
with open(dir_structure_json_path, "w") as f:
    json.dump(structure, f, indent=4)
print(f"Save dir structure to {dir_structure_json_path}")

# split train, val
train_imgs, val_imgs = split_train_val_data(structure)
print("Number of train imgs: ", len(train_imgs))
print("Number of val imgs: ", len(val_imgs))

# get_pairs
num_per_class = 100
scale_factor = [1, 0.2, 0.2]
print("Generating train: ")
output_train = generate_data(structure, train_imgs, num_per_class, scale_factor)

num_per_class = 20
print("Generating val: ")
output_val = generate_data(structure, val_imgs, num_per_class, scale_factor)

with open(cap_train_path, 'w') as f:
    json.dump(output_train, f, indent=4)
with open(cap_val_path, 'w') as f:
    json.dump(output_val, f, indent=4)

