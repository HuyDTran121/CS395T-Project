import os
import shutil
folder = 'few-shot-images/CUB_200_2011/CUB_200_2011/images'

subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

for subfolder in subfolders:
    final_path = "few-shot-images/birds"
    shutil.copy(os.path.join(subfolder, os.listdir(subfolder)[0]), os.path.join(final_path, os.listdir(subfolder)[0]))