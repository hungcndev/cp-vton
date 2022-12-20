import os
import glob
import wget
from tqdm import tqdm

if __name__ == "__main__":
    root = "/data/DataSet/UPT/UPT_train_url_list.txt"
    os.makedirs("/data/DataSet/UPT/UPT/train", exist_ok=True)
    count = 1
    with open(root, "r") as f:
        while True:
            line = f.readline()

            if not line: break
            
            wget.download(line, out="/data/DataSet/UPT/UPT/train/")