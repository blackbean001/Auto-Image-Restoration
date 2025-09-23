import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector

import json
import os
import shutil
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms.functional as FF
from torch.utils.data import DataLoader

import clip
from clip.model import CLIP
import PIL
import PIL.Image
from tqdm import tqdm
import time


# define device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return FF.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FF.pad(image, padding, 0, 'constant')


# element_wise_sum function
def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


# Combiner class
class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



# define DataClass
class ImgResDataset(Dataset):
    def __init__(self, preprocess: callable, imgs_path: list, mode='classic', phase='val'):
        self.preprocess = preprocess
        self.imgs_path = imgs_path
        self.mode = mode
        self.phase = phase

        print(f"Loading ImgRes images...")

    def __getitem__(self, index):
        if self.mode == "classic":
            image_path = self.imgs_path[index]
            image = self.preprocess(PIL.Image.open(image_path))
            return image_path, image
    
    def __len__(self):
        if self.mode == "classic":
            return len(self.imgs_path)

        

# create connections
def create_connection():
    # connect to PostgreSQL
    try:
        conn = psycopg2.connect(dbname="agenticir_rag_test",\
                                user="postgres",\
                                host="/var/run/postgresql")
        print("Successfully connect to PostgreSQL ! ")
    except psycopg2.Error as e:
        print(f"Connection to PostgreSQL failed: {e}")

    # register with pgvector
    try:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
    except psycopg2.Error as e:
        print(f"pgvector registration failed: {e} ")
    
    return conn


# execute query
def execute_query(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        conn.commit()
        print("Execution success !")
    except psycopg2.Error as e:
        print(f"Execution failed：{e}")
    return cursor


# create table if not exist
def init_table(conn):
    # create table
    query = """
    CREATE TABLE IF NOT EXISTS ImgresEmbedding (
        id SERIAL,
        name VARCHAR(255) PRIMARY KEY,
        res_seq VARCHAR(500) NOT NULL,
        embedding vector(640)
    );
    """
    # execute query
    if conn:
        _ = execute_query(conn, query)


# close connection
def close_connection(conn):
    if conn:
        conn.close()
        print("PostgreSQL closed...")


# use clip model to generate embedding for retrieval
def extract_index_features(dataset, clip_model):
    feature_dim = clip_model.visual.output_dim
    def collate_fn(batch):
        """
        Discard None images in a batch when using torch DataLoader
        :param batch: input_batch
        :return: output_batch = input_batch - None_values
        """
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, \
            num_workers=multiprocessing.cpu_count(),pin_memory=True, collate_fn=collate_fn)

    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)

    return index_features, index_names


# generate embeddings
def insert_to_postgresql(conn, clip_model, \
        imgres_dir=Path("/home/jason/AgenticIR/output").resolve()):
    
    img_path_list = []
    res_seqs = []
    for img_file in os.listdir(imgres_dir):
        summary_path = imgres_dir / img_file / "logs" / "summary.json"
        if not os.path.exists(summary_path):
            print(f"WARNING: THE IMAGE FILES IN {summary_path} IS NOT COMPLETE, PLEASE CHECK !!!")
            shutil.rmtree(imgres_dir / img_file)
            continue
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        subtasks = summary["execution_path"]["subtasks"]
        tools = summary["execution_path"]["tools"]
        res_seqs.append("/".join([subtasks[i]+"_"+tools[i] for i in range(len(tools))]))
        img_path_list.append(summary["tree"]["img_path"])
    
    #img_path_list = [item.replace("output", "output.old") for item in img_path_list]

    classic_test_dataset = ImgResDataset(preprocess, img_path_list, 'classic', 'val')
    index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
    print("Number of images to insert:", len(img_path_list), len(index_names))
    # print(index_names[0])  # /home/jason/AgenticIR/output/low resolution+motion blur+rain_003.png-250908_203929/img_tree/0-img/input.pn
    # print(index_features[0]) # dim: 640
    
    # insert into postgresql
    insert_data_list = []
    for (name, res_seq, emb) in zip(index_names, res_seqs, index_features):
        insert_data_list.append((name, res_seq, emb.cpu().numpy()))
    
    #insert_cmd = f"INSERT INTO ImgresEmbedding (name, res_seq, embedding) \
    #            VALUES {insert_data_list}"
    
    if conn:
        cur = conn.cursor()
        extras.execute_values(cur, "INSERT INTO ImgresEmbedding (name, res_seq, embedding) VALUES %s \
                ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding", insert_data_list)
    
    # count number of records
    count_cmd = "SELECT count(*) FROM ImgresEmbedding"
    if conn:
        cur = execute_query(conn, count_cmd)

    print(f"Finished inserting {len(insert_data_list)} new records!!! Table now updated to {cur.fetchone()[0]}.")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--imgs-dir", type=str, required=True, help="Directory for input images, structure defined in ImgResDataset")
    parser.add_argument("--combiner-path", type=str, help="path to trained Combiner")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--transform", default="targetpad", type=str, \
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")

    args = parser.parse_args()

    # create connection
    conn = create_connection()
    
    # create table if not exist
    init_table(conn)

    # preparation works
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim    
    
    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device)
        saved_state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(saved_state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    # generate embeddings
    insert_to_postgresql(conn, clip_model, imgres_dir=Path(args.imgs_dir).resolve())





