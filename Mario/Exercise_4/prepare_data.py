from collections import defaultdict
from pathlib import Path
import multiprocessing
from tqdm.contrib.concurrent import process_map  # or thread_map
from torchvision.transforms import Normalize
import numpy as np
import scipy
import torch
import random
from PIL import Image
from scipy import ndimage
from torch import Tensor

from typing import Dict, List, Tuple
import urllib.request
urllib = getattr(urllib, 'request', urllib)
from tqdm import tqdm
import zipfile
from os import remove


# Original number of expected files
_NUM_EXPECTED_FILES_BACKGROUND = -1
_NUM_EXPECTED_FILES_EVAL = -1
_NUM_ROTATIONS = 4

_EVAL_URL = "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
_BG_URL = "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"

_BG_DIR_NAME = "images_background"
_EVAL_DIR_NAME = "images_evaluation"
_TORCH_NORM = Normalize([0.5], [0.5])

# From https://github.com/tqdm/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


class ImagePreProcessor:
    """
    A class which can be used in multi-processing to convert our images.
    Images are downsanmpled and rotated. Rotated images are saved into a new label directory in which the rotation
    is appended to the dir name
    """
    def __init__(self, prep_path: Path, img_size=(28,28), rotations=(0, 90, 180, 270)):
        self.prep_path = prep_path
        self.img_size = img_size
        self.rotations = rotations

    def __call__(self, image_path: Path):
        char_dir = image_path.parent
        alphabet_dir = char_dir.parent

        # Read the data as grayscale image and downscale it to (28 x 28) pixels
        img_data_og = Image.open(image_path).convert('L')
        img_downscaled = img_data_og.resize(self.img_size)
        # Images are rotated four times
        for rotation in self.rotations:
            # Create a new directory in the prep directory;
            # We save the rotation in the labels and the file names
            new_img_file_name = image_path.stem + f"_r{rotation}" + image_path.suffix
            new_char_dir = char_dir.name + f"_r{rotation}"
            new_img_path = self.prep_path / alphabet_dir.name / new_char_dir / new_img_file_name
            new_img_path.parent.mkdir(exist_ok=True, parents=True)
            if new_img_path.exists():
                continue

            # Rotate the image
            img_rotated = img_downscaled.rotate(rotation)
            img_rotated.save(new_img_path, "png")


def sample_from_torch_sample_dict(class_list: List[str],
                                  class_to_sample_dict: Dict[str, List[Tensor]],
                                  num_classes: int, num_support: int, num_query: int) \
        -> Tuple[Tensor, Tensor, List[str]]:
    classes = list(random.sample(class_list, k=num_classes))
    # get the shape of some data sample
    data_shape = class_to_sample_dict[classes[0]][0].shape
    support_set = torch.empty((num_classes, num_support, *data_shape))
    query_set = torch.empty((num_classes, num_query, *data_shape))

    for class_idx_k in range(num_classes):
        class_samples = class_to_sample_dict[classes[class_idx_k]]
        random.shuffle(class_samples)
        support_set[class_idx_k] = torch.stack(class_samples[:num_support])
        query_set[class_idx_k] = torch.stack(class_samples[num_support:(num_support+num_query)])

    return support_set, query_set, classes


class OmniGlotDataSet:

    _META_KEY = "__meta__"
    _DATA_DIM = (1, 28, 28)

    def __init__(
            self,
            root_path: Path,
    ):
        self.root_path = root_path
        self.raw_path = root_path / "omniglot_raw"
        self.prep_path = root_path / "omniglot_prep"

    def download_and_unzip_omniglot(self):
        def _download_url(url: str, dest: Path):
            print(f"Downloading: {url}: {dest}")
            dest.parent.mkdir(exist_ok=True, parents=True)
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                          desc=url.split('/')[-1]) as t:  # all optional kwargs
                urllib.urlretrieve(url, filename=str(dest.absolute()),
                                   reporthook=t.update_to, data=None)
                t.total = t.n

        bg_zip_path = self.raw_path / _BG_URL.split("/")[-1]
        eval_zip_path = self.raw_path / _EVAL_URL.split("/")[-1]
        # Download zips
        _download_url(_BG_URL, bg_zip_path)
        _download_url(_EVAL_URL, eval_zip_path)
        # Unzip the files
        for omni_zip_file in [bg_zip_path, eval_zip_path]:
            print(f"Unzipping: {omni_zip_file}")
            with zipfile.ZipFile(omni_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.raw_path)

            print(f"Deleting {omni_zip_file}")
            remove(omni_zip_file)

    def prepare_data_set(self, force_override=False):
        print("Starting to prepare Omniglot dataset")
        if not force_override and self._check_data_set_is_processed():
            print("Data pre-processing started...")
            print("Data has been processed and force_override=False...skipping pre-processing")
            return
        # Rotate images
        if not self.raw_path.exists() or force_override:
            self.download_and_unzip_omniglot()
        else:
            print("Skipping downloading Omniglot as original files already exist..")
        if not self.prep_path.exists() or force_override:
            self.pre_process_omniglot()
        else:
            print("Skipping pre-processing as prep dirs already exist...")

    def pre_process_omniglot(self):
        def prep_data_part(raw_data_part_dir: Path, prep_part_dir: Path):
            print(f"Preparing data set part: {raw_data_part_dir}")
            image_files = list(raw_data_part_dir.glob("*/*/*.png"))
            img_processor = ImagePreProcessor(prep_path=prep_part_dir)
            r = process_map(img_processor, image_files, max_workers=multiprocessing.cpu_count(), chunksize=512)

        print("Starting pre-processing")
        prep_data_part(self.raw_path / _BG_DIR_NAME, self.prep_path / _BG_DIR_NAME)
        prep_data_part(self.raw_path / _EVAL_DIR_NAME, self.prep_path / _EVAL_DIR_NAME)
        print(f"Finished pre-processing images...see {self.prep_path}")

    def _check_data_set_is_processed(self):
        if not self.prep_path.exists():
            return False

        bg_path = self.prep_path / _BG_DIR_NAME
        eval_path = self.prep_path / _EVAL_DIR_NAME

        for p in bg_path, eval_path:
            if not p.exists():
                return False

        num_prep_bg = len(list(bg_path.glob("*/*.png")))
        if num_prep_bg != _NUM_ROTATIONS * _NUM_EXPECTED_FILES_BACKGROUND:
            return False
        num_prep_eval = len(list(eval_path.glob("*/*.png")))
        if num_prep_eval != _NUM_ROTATIONS * _NUM_EXPECTED_FILES_EVAL:
            return False

        return True

    def load_image_to_tensor(self, img_path: Path) -> Tensor:
        np_image = np.array(Image.open(img_path).convert("L"))
        image_tensor = torch.unsqueeze(torch.Tensor(np_image), 0)  # Load image as tensor of shape (1, 28, 28)
        image_tensor /= 255  # Scale to [0, 1]
        image_tensor = _TORCH_NORM(image_tensor)  # normalize
        return image_tensor

    def load_image_and_label(self, image_path: Path) -> Tuple[Tensor, str]:
        char_index = image_path.parent.name
        alphabet = image_path.parents[1].name
        label = f"{alphabet}__{char_index}"
        img_tensor = self.load_image_to_tensor(image_path)
        return img_tensor, label

    def create_example_img_files_dict(self,
                                      part="background",
                                      num_classes: int = 10,
                                      num_support: int = 5, num_query: int = 5):

        data_dir = self.get_prep_data_part(part)
        labels = []
        for alphabet_dir in data_dir.glob("*"):
            if not alphabet_dir.is_dir():
                continue
            for char_dir in alphabet_dir.glob("*"):
                if not char_dir.is_dir():
                    continue
                labels.append(f"{alphabet_dir.name}__{char_dir.name}")

        sampled_labels = list(sorted(random.sample(labels, k=num_classes)))
        label_to_img_names = {OmniGlotDataSet._META_KEY: {"part": part, "num_classes": num_classes,
                                                          "num_support": num_support, "num_query": num_query},
                              "labels": {}}
        for label in sampled_labels:
            label_to_img_names["labels"][label] = {}
            #Get all images for this label
            alphabet_dir_name, char_name = label.split("__")
            char_dir = data_dir / alphabet_dir_name / char_name
            random_img_paths = list(char_dir.glob("*.png"))

            # get random support and query images
            random.shuffle(random_img_paths)
            support_imgs = random_img_paths[:num_support]
            query_imgs = random_img_paths[num_support:(num_support+num_query)]

            # Write the file names of the support and query images to the dict
            for part_name, part_imgs in zip(("support", "query"), (support_imgs, query_imgs)):
                label_to_img_names["labels"][label][part_name] = [img.name for img in part_imgs]

        return label_to_img_names

    def get_prep_data_part(self, part: str) -> Path:
        if part == "background":
            data_dir = self.prep_path / _BG_DIR_NAME
        elif part == "evaluation":
            data_dir = self.prep_path / _EVAL_DIR_NAME
        else:
            raise ValueError(f"Could not match data part for the omniglot data set {part}."
                             f"Only 'background' and 'evaluation' are supported")
        return data_dir

    def load_example_label_to_img_dict(self, label_to_img_dict) -> Tuple[Tensor, Tensor, List[str]]:
        """
        Loads the label_to_img_dict, containing labels as keys and file names of
        In the _meta_ key, the data file part and num classes is specified
        """
        meta_dict = label_to_img_dict[OmniGlotDataSet._META_KEY]
        part = meta_dict["part"]
        data_dir = self.get_prep_data_part(part)
        num_classes = meta_dict["num_classes"]
        num_support = meta_dict["num_support"]
        num_query = meta_dict["num_query"]

        support_batch = torch.empty((num_classes, num_support, *OmniGlotDataSet._DATA_DIM))
        query_batch = torch.empty((num_classes, num_query, *OmniGlotDataSet._DATA_DIM))

        label_idx = 0
        labels_list = []
        for label_idx, label in enumerate(label_to_img_dict["labels"]):
            labels_list.append(label)

            # Load support and query images
            alphabet, char_name = label.split("__")
            char_dir = data_dir / alphabet / char_name

            for img_idx, img_name in enumerate(label_to_img_dict["labels"][label]["support"]):
                img_path = char_dir / img_name
                img_tensor = self.load_image_to_tensor(img_path)
                support_batch[label_idx][img_idx] = img_tensor

            for img_idx, img_name in enumerate(label_to_img_dict["labels"][label]["query"]):
                img_path = char_dir / img_name
                img_tensor = self.load_image_to_tensor(img_path)
                query_batch[label_idx][img_idx] = img_tensor

        return support_batch, query_batch, labels_list

    def init_torch_sample_dict(self, part="background", _debug: bool = False, use_multi_proc: bool = False) -> Dict[str, List[Tensor]]:
        data_dir = self.get_prep_data_part(part)

        print(f"Loading all image data of data set part {part}")
        img_paths = list(data_dir.glob("*/*/*.png"))
        if _debug:
            print("WARNING: Limiting img_paths due to DEBUG")
            img_paths = img_paths[:1000]
        if use_multi_proc:
            res = process_map(self.load_image_and_label, img_paths,  max_workers=multiprocessing.cpu_count(),
                             total=len(img_paths), chunksize=1024)
        else:
            res = []
            for img_path in tqdm(img_paths, total=len(img_paths)):
                res.append(self.load_image_and_label(img_path))
        label_to_data = defaultdict(list)
        for img_t, label in res:
            label_to_data[label].append(img_t)
        return label_to_data


if __name__ == "__main__":
    root_dir = Path(__file__).parent / "data"
    omniglot_dataset = OmniGlotDataSet(root_dir)
    omniglot_dataset.prepare_data_set()
    bg_label_to_img = omniglot_dataset.init_torch_sample_dict(part="background")
    eval_label_to_img = omniglot_dataset.init_torch_sample_dict(part="evaluation")

