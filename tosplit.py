import os, shutil, random
from glob import glob

IMAGE_DIR = r"C:\Users\babuk\Downloads\helmet with label\helmet"
OUTPUT = r"C:\Users\babuk\Documents\HOPE AI\Deep Learning\Week11-Deep Learning Module\Bike Helmet Detection\HelmetDataset"

os.makedirs(f"{OUTPUT}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT}/images/valid", exist_ok=True)
os.makedirs(f"{OUTPUT}/images/test", exist_ok=True)

os.makedirs(f"{OUTPUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT}/labels/valid", exist_ok=True)
os.makedirs(f"{OUTPUT}/labels/test", exist_ok=True)

images = glob(f"{IMAGE_DIR}/*.jpg") + glob(f"{IMAGE_DIR}/*.png")
random.shuffle(images)

train_end = int(len(images) * 0.7)
valid_end = int(len(images) * 0.9)

train_imgs = images[:train_end]
valid_imgs = images[train_end:valid_end]
test_imgs = images[valid_end:]

def move_files(file_list, split):
    for img in file_list:
        label = img.replace(".jpg", ".txt").replace(".png", ".txt")
        shutil.copy(img, f"{OUTPUT}/images/{split}/")
        shutil.copy(label, f"{OUTPUT}/labels/{split}/")

move_files(train_imgs, "train")
move_files(valid_imgs, "valid")
move_files(test_imgs, "test")

print("Dataset split completed.")
