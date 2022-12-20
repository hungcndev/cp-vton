import torch
from torchvision import transforms

from glob import glob
from typing import Optional, Callable, List
from PIL import Image


'''
root: "../FASCODE_IMAGE" 

# FASCODE_IMAGE file tree
FASCODE_IMAGE/
└──image
    ├── BL-001.jpg
    ├── BL-002.jpg
    ├── BL-003.jpg
    ├── BL-004.jpg
    ├── ...
    └── VT-111.jpg
# FASCODE_IMAGE classification
outer
├── JK(자켓)
├── JP(점퍼)
├── CT(코트)
├── CD(가디건)
└── VT(조끼)
top
├── KN(니트)
├── SW(스웨터)
├── SH(셔츠)
└── BL(블라우스)
bottom
├── SK(치마)
├── PT(바지)
└── OP(원피스)
shoe
└── SE(신발)
# How to use DataLoader
FASCODE_IMAGE(
    root="../FASCODE_IMAGE/",
    categories="outer" or "top" or "bottom" or "shoe",
    item="JK" or "JP" or ... or "SE"
)
'''

def get_img_path(root: str, categories: Optional[List[str]] = None, item : Optional[List[str]] = None):
            
    if root[-1] != "/":
        root = root + "/"

    img_path = []
    if categories is not None:
        for cat in categories:
            if cat == "outer":
                img_path += glob(root + "image/JK*.jpg") + glob(root + "image/JP*.jpg") + glob(root + "image/CT*.jpg") + glob(root + "image/CD*.jpg") + glob(root + "image/VT*.jpg")
            elif cat == "top":
                img_path += glob(root + "image/KN*.jpg") + glob(root + "image/SW*.jpg") + glob(root + "image/SH*.jpg") + glob(root + "image/BL*.jpg")
            elif cat == "bottom":
                img_path += glob(root + "image/SK*.jpg") + glob(root + "image/PT*.jpg") + glob(root + "image/OP*.jpg")
            elif cat == "shoe":
                img_path += glob(root + "image/SE*.jpg")
    elif item is not None:
        for i in item:
            img_path += glob(root + f"image/{item}*.jpg")
    else:
        img_path = glob(root + "image/*.jpg")

    return img_path

class FASCODE_IMAGE(torch.utils.data.Dataset):

    def __init__(
        self, 
        root: str, 
        categories: Optional[List[str]] = None, 
        item : Optional[List[str]] = None, 
        transform: Optional[Callable] = None
        ):

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 192)),
                transforms.Normalize((0.5, ), (0.5, )),
            ])

        self.transform = transform
        self.img_path = get_img_path(root, categories, item)
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]

        img = Image.open(img_path).convert("RGB")

        img_transformed = self.transform(img)

        label = img_path.split('/')[-1].split('-')[0]

        return img_transformed, label