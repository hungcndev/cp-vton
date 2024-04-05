import os
import tarfile
import shutil
import gdown


url = "https://drive.google.com/uc?export=download&confirm=CONFIRM&id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo"
output = "data/viton_resize.tar.gz"
gdown.download(url, output, quiet=False)

tarfile.open(output).extractall(path='data/')

shutil.move('data/viton_resize/test/', 'data/test/')
shutil.move('data/viton_resize/train/', 'data/train/')

os.rmdir('data/viton_resize/')
os.remove('data/viton_resize.tar.gz')

