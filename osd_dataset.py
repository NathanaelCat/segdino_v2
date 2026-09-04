"""OSD four-class dataset adapter for the standalone SegDINO-v2 runner."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _make_pairs(root: Path, split: str) -> list[tuple[Path, Path, str]]:
    image_dir = root / split / "images"
    mask_dir = root / split / "masks_indexed"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"OSD image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"OSD indexed-mask directory not found: {mask_dir}")

    images = {path.stem: path for path in _image_files(image_dir)}
    masks = {path.stem: path for path in _image_files(mask_dir)}
    if set(images) != set(masks):
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        raise ValueError(
            f"OSD {split} image/mask IDs do not match; "
            f"missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )
    return [(images[key], masks[key], key) for key in sorted(images)]


class OSDTransform:
    """Resize an image/mask pair while preserving indexed mask labels.

    The default augmentation follows the public SegDINO training script's
    lightweight profile.  Every operation is explicit in the JSON config so
    an OSD run can be reproduced without inheriting hidden torchvision state.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        train: bool = False,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        rotate_degrees: int = 0,
        color_jitter: Optional[dict] = None,
        resize_mode: str = "stretch",
        ignore_index: int = 255,
    ) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(int(value) for value in size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError(f"Invalid image size: {size}")
        self.train = train
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_degrees = int(rotate_degrees)
        self.color_jitter = ColorJitter(**(color_jitter or {})) if color_jitter else None
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError("resize_mode must be 'stretch' or 'letterbox'")
        self.resize_mode = resize_mode
        self.ignore_index = int(ignore_index)

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.resize_mode == "stretch":
            return (
                TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR),
                TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST),
            )

        target_h, target_w = self.size
        source_w, source_h = image.size
        scale = min(target_h / source_h, target_w / source_w)
        resized_h = max(1, min(target_h, round(source_h * scale)))
        resized_w = max(1, min(target_w, round(source_w * scale)))
        image = TF.resize(image, [resized_h, resized_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [resized_h, resized_w], interpolation=InterpolationMode.NEAREST)
        pad_left = (target_w - resized_w) // 2
        pad_right = target_w - resized_w - pad_left
        pad_top = (target_h - resized_h) // 2
        pad_bottom = target_h - resized_h - pad_top
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        image = TF.pad(image, padding, fill=0)
        # Letterbox pixels are not part of the source image and must not alter
        # the global confusion matrix through artificial background pixels.
        mask = TF.pad(mask, padding, fill=self.ignore_index)
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._resize(image, mask)

        if self.train and self.horizontal_flip and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.train and self.vertical_flip and random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if self.train and self.rotate_degrees > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=self.ignore_index,
            )
        if self.train and self.color_jitter is not None and random.random() < 0.5:
            image = self.color_jitter(image)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        # Do not use TF.to_tensor for the mask: it converts labels to floats
        # and scales 1/2/3 as if they were grayscale intensities.
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return image_tensor, mask_tensor


class OSDDataset(Dataset):
    """Four-class OSD dataset using ``images`` and ``masks_indexed``."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[OSDTransform] = None,
        num_classes: int = 4,
        ignore_index: int = 255,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.pairs = _make_pairs(self.root, split)
        self.transform = transform
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, mask_path, sample_id = self.pairs[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()

        valid = mask_tensor != self.ignore_index
        if valid.any():
            min_value = int(mask_tensor[valid].min())
            max_value = int(mask_tensor[valid].max())
            if min_value < 0 or max_value >= self.num_classes:
                raise ValueError(
                    f"OSD mask {mask_path} has labels [{min_value}, {max_value}], "
                    f"expected 0..{self.num_classes - 1} or {self.ignore_index}"
                )
        return image_tensor, mask_tensor, sample_id


def build_osd_transform(
    size: int | Sequence[int],
    train: bool,
    augment: Optional[dict] = None,
    resize_mode: str = "stretch",
    ignore_index: int = 255,
) -> OSDTransform:
    augment = augment or {}
    jitter = augment.get("color_jitter", {}) if train else None
    return OSDTransform(
        size=size,
        train=train,
        horizontal_flip=bool(augment.get("horizontal_flip", False)),
        vertical_flip=bool(augment.get("vertical_flip", False)),
        rotate_degrees=int(augment.get("rotate_degrees", 0)),
        color_jitter=jitter,
        resize_mode=resize_mode,
        ignore_index=ignore_index,
    )
