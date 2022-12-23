import os
import random
'''
목표 : test_pairs.txt에 없는 파일 삭제하기
과정 : test_pairs.txt
'''
pairs_list = "data/test_pairs.txt"
cloth_path = "data/test/cloth"
body_path = "data/test/image"

list = ""
cloth_list = os.listdir(cloth_path)
body_list = os.listdir(body_path)
random.shuffle(body_list)

for idx, cloth in enumerate(cloth_list):
    list += body_list[idx] + " " + cloth_list[idx] + "\n"

with open(pairs_list, "w") as f:
    f.write(list)