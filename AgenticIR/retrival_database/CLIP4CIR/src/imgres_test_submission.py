import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from combiner_train import extract_index_features
from data_utils import CIRRDataset, ImgResDataset, targetpad_transform, squarepad_transform, base_path
from combiner import Combiner
from utils import element_wise_sum, device


def generate_imgres_test_submissions(combining_function: callable, file_name: str, clip_model: CLIP,
                                   preprocess: callable):
    """
   Generate and save ImgRes test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
   :param file_name: file_name of the submission
   :param clip_model: CLIP model
   :param preprocess: preprocess pipeline
   """

    clip_model = clip_model.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = ImgResDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
    #print(len(index_features), len(index_names))
    relative_test_dataset = ImgResDataset('val', 'relative', preprocess)

    # Generate test prediction dicts for ImgRes
    pairid_to_predictions = generate_imgres_test_dicts(relative_test_dataset, clip_model, index_features, index_names, combining_function)
    submission = {
        'version': 'test_jason',
        'metric': 'recall'
    }

    submission.update(pairid_to_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'ImgRes'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving ImgRes test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, indent=4, sort_keys=True)


def generate_imgres_test_dicts(relative_test_dataset: ImgResDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for ImgRes dataset
    :param relative_test_dataset: ImgRes test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_features, reference_names, pairs_id, all_captions = \
        generate_imgres_test_predictions(clip_model, relative_test_dataset, combining_function, index_names, index_features)

    print(f"Compute ImgRes prediction dicts")
    #for i in range(len(pairs_id)):
    #    print(reference_names[i], pairs_id[i], all_captions[i])
    #print(len(reference_names), len(pairs_id), len(all_captions), len(index_features))
    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()
    print("feature size: ", predicted_features.shape, index_features.shape)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                sorted_index_names.shape[1] - 1)
    #print("all_captions: ", all_captions)
    # Generate prediction dicts
    paired_to_predictions = {}
    pairs_id_count = defaultdict(int)

    for i in range(len(pairs_id)):
        paired_to_predictions[pairs_id[i]+"_"+str(pairs_id_count[pairs_id[i]])] = [all_captions[i], sorted_index_names[i][:10].tolist()]
        pairs_id_count[pairs_id[i]] += 1
    #paired_to_predictions = {pair_id: prediction[:10].tolist() for (pair_id, prediction) in
    #                         zip(pairs_id, sorted_index_names)}

    return paired_to_predictions


def generate_imgres_test_predictions(clip_model: CLIP, relative_test_dataset: ImgResDataset, combining_function: callable,
                                   index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute ImgRes predictions on the test set
    :param clip_model: CLIP model
    :param relative_test_dataset: ImgRes test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute ImgRes test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))
    
    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    reference_names = []
    
    all_captions = []

    for batch_pairs_id, batch_reference_names, captions in tqdm(
            relative_test_loader):  # Load data
        
        t0 = time.time()
        captions = list(captions[0])

        all_captions.extend(captions)

        text_inputs = clip.tokenize(captions, context_length=77).to(device)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            #if text_features.shape[0] == 1:
            #    reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            #else:
            #    reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
            #        name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_pairs_id)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_pairs_id)(
                    name_to_feat))

            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)
        t1 = time.time()
        print("Processing time for each batch: ", t1 - t0)
    return predicted_features, reference_names, pairs_id, all_captions


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=str, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    args = parser.parse_args()
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
    
    t0 = time.time()
    generate_imgres_test_submissions(combining_function, args.submission_name, clip_model, preprocess)
    t1 = time.time()
    print("Total retrieval time: ", t1 - t0)

if __name__ == '__main__':
    main()
