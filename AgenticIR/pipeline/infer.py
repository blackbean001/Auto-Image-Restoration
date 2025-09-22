from pathlib import Path
import random
from .iragent import IRAgent
import os

#input_path = Path("/home/jason/UVDoc/collected_data/30.png").resolve()
#input_path = Path("/home/jason/CLIP4Cir/ImgRestore_dataset/images/LQ/d3/low resolution+dark+rain/095.png").resolve()
#input_path = Path("/home/jason/AgenticIR/output/009-250826_211322/img_tree/0-img/input.png").resolve()
#input_path = Path("/home/jason/UVDoc/UVDoc_benchmark/img/00048_unwarp.png").resolve()
#input_path = Path("/home/jason/UVDoc/collected_data/32_blur.jpg").resolve()
input_path = Path("/home/jason/UVDoc/collected_data/g_2_blur_resize.png").resolve()
output_dir = Path("/home/jason/AgenticIR/output").resolve()

# get all files

def get_all_files(dir_path):
    file_name_list = []
    for root, dirs, files in os.walk(dir_path):
        if files:
            for name in files:
                file_name = '{0}/{1}'.format(root, name).replace('\\', '/')
                file_name_list.append(file_name)
    return file_name_list

"""
input_path = Path(input_path).resolve()
agent = IRAgent(
    input_path=input_path, output_dir=output_dir,
    evaluate_degradation_by="depictqa",
    #evaluate_degradation_by="clip_retrieval",
    with_retrieval=True,
    with_reflection=True,
    reflect_by="depictqa",
    with_rollback=True,
    silent=False
)

agent.run()
"""


dir_path = "/home/jason/AgenticIR/retrival_database/LQ"
input_img_list = get_all_files(dir_path)
input_img_list = [item for item in input_img_list if ("img_tree" not in item and "logs" not in item)]
random.shuffle(input_img_list)
print("Total number of images to process: ", len(input_img_list))

for i,path in enumerate(input_img_list[:500]):
    try:
        print(f"Processing {i}/{len(input_img_list)}", path)
        input_path = Path(path).resolve()
        agent = IRAgent(
            input_path=input_path, output_dir=output_dir,
            evaluate_degradation_by="depictqa",
            #evaluate_degradation_by="clip_retrieval",
            with_retrieval=True,
            with_reflection=True,
            reflect_by="depictqa",
            with_rollback=True,
            silent=False
        )

        agent.run()
    except:
        print(f"Failed for {path}")

