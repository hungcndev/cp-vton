import os
import tarfile
import shutil
import gdown

# Download and prepare dataset for training and testing

url = "https://drive.google.com/uc?export=download&confirm=CONFIRM&id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo"
output = "data/viton_resize.tar.gz"
gdown.download(url, output, quiet=False)

tarfile.open(output).extractall(path='data/')

shutil.move('data/viton_resize/test/', 'data/test/')
shutil.move('data/viton_resize/train/', 'data/train/')

os.rmdir('data/viton_resize/')
os.remove('data/viton_resize.tar.gz')


# Download and prepare checkpoints for GMM and TOM models

gmm_url = "https://drive.google.com/file/d/1lGPLTDEuRgYvdJIgS4L0qeAp7LQn6hIr/view"
gmm_file = "gmm_final.pth"
gdown.download(gmm_url, gmm_file, quiet=False)
shutil.move(gmm_file, 'checkpoints/GMM/')


tom_url = "https://drive.google.com/file/d/10wnB1coU73u3rbRgevHRyadO7b1tCron/view"
tom_file = "tom_final.pth"
gdown.download(tom_url, tom_file, quiet=False)
shutil.move(tom_file, 'checkpoints/TOM/')
