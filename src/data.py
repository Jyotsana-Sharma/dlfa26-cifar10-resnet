"""Rule-compliant competition PNG loading and deterministic stratified splits."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from functools import lru_cache
from dataclasses import dataclass
import io
import os
import zipfile
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

@dataclass(frozen=True)
class ZipImagePath:
    """A lazy ZIP member; exposes the relevant pathlib API for provenance hashes."""
    archive: Path
    member: str

    @property
    def name(self) -> str:
        return Path(self.member).name

    def resolve(self) -> str:
        return f"{self.archive.resolve()}::{self.member}"

    def read_bytes(self) -> bytes:
        return _archive_handle(self.archive).read(self.member)

    def __str__(self) -> str:
        return self.resolve()


@lru_cache(maxsize=8)
def _archive_index(path: str) -> dict[str, str]:
    """Index only real PNG/JPEG files, avoiding macOS metadata and duplicates."""
    with zipfile.ZipFile(path) as archive:
        index = {}
        for info in archive.infolist():
            p = Path(info.filename)
            if info.is_dir() or '__MACOSX' in p.parts or p.name.startswith('._'):
                continue
            if p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if p.name in index:
                raise ValueError(f"Duplicate image basename in {path}: {p.name}")
            index[p.name] = info.filename
    return index


_zip_handles: dict[tuple[int, str], zipfile.ZipFile] = {}


def _archive_handle(path: Path) -> zipfile.ZipFile:
    # Each DataLoader worker gets a separate handle, avoiding shared ZIP seeks.
    key = (os.getpid(), str(path.resolve()))
    handle = _zip_handles.get(key)
    if handle is None:
        handle = zipfile.ZipFile(path)
        _zip_handles[key] = handle
    return handle


def _open_rgb(path: Path | ZipImagePath) -> Image.Image:
    if isinstance(path, ZipImagePath):
        with Image.open(io.BytesIO(path.read_bytes())) as image:
            return image.convert('RGB')
    with Image.open(path) as image:
        return image.convert('RGB')



@dataclass(frozen=True)
class LabeledImage:
    image_id: str
    path: Path | ZipImagePath
    label: str


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def infer_columns(rows: Sequence[dict[str, str]]) -> tuple[str, str]:
    columns = list(rows[0])
    lowered = {name.lower(): name for name in columns}
    id_column = next(
        (lowered[name] for name in ("id", "image_id", "image", "filename", "file") if name in lowered),
        None,
    )
    label_column = next(
        (lowered[name] for name in ("label", "class", "target") if name in lowered), None
    )
    if id_column is None or label_column is None:
        raise ValueError(f"Could not infer id/label columns from {columns}")
    return id_column, label_column


def resolve_image(image_dir: Path, image_id: str) -> Path | ZipImagePath:
    if not image_id or Path(image_id).name != image_id:
        raise ValueError("Image filenames must be nonempty basenames")
    if image_dir.is_file() and image_dir.suffix.lower() == '.zip':
        index = _archive_index(str(image_dir.resolve()))
        options = [image_id] + [f"{image_id}{suffix}" for suffix in sorted(IMAGE_SUFFIXES)]
        matches = [index[candidate] for candidate in options if candidate in index]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one image for {image_id!r} in {image_dir}")
        return ZipImagePath(image_dir.resolve(), matches[0])
    direct = image_dir / image_id
    if direct.is_file():
        return direct
    matches = [image_dir / f"{image_id}{suffix}" for suffix in sorted(IMAGE_SUFFIXES)]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one image for id {image_id!r} in {image_dir}")
    return matches[0]


def load_labeled_images(labels_csv: str | Path, image_dir: str | Path) -> list[LabeledImage]:
    rows = read_csv(labels_csv)
    id_column, label_column = infer_columns(rows)
    image_dir = Path(image_dir)
    file_column = "filename" if "filename" in rows[0] else id_column
    samples = [
        LabeledImage(row[id_column], resolve_image(image_dir, row[file_column]), row[label_column])
        for row in rows
    ]
    if len(samples) != 50_000:
        raise ValueError(f"Expected 50,000 permitted training images, found {len(samples):,}")
    if len({sample.image_id for sample in samples}) != len(samples):
        raise ValueError("Duplicate training image IDs found")
    if len({sample.path.resolve() for sample in samples}) != len(samples):
        raise ValueError("Duplicate training image filenames found")
    return samples


def class_mapping(samples: Sequence[LabeledImage]) -> tuple[dict[str, int], list[str]]:
    observed = {sample.label for sample in samples}
    expected = {str(index) for index in range(10)}
    if observed != expected:
        raise ValueError(f"Expected integer class labels 0 through 9, found {sorted(observed)}")
    labels = [str(index) for index in range(10)]
    return {label: index for index, label in enumerate(labels)}, labels


def stratified_split(
    samples: Sequence[LabeledImage], validation_size: int = 5_000, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Dependency-free deterministic split with exact proportional allocation."""
    import random

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_label[sample.label].append(index)
    if not 0 < validation_size < len(samples):
        raise ValueError("validation_size must be smaller than the dataset")
    rng = random.Random(seed)
    validation: list[int] = []
    remaining = validation_size
    labels = sorted(by_label)
    for position, label in enumerate(labels):
        indices = by_label[label]
        rng.shuffle(indices)
        if position == len(labels) - 1:
            count = remaining
        else:
            count = round(validation_size * len(indices) / len(samples))
            remaining -= count
        validation.extend(indices[:count])
    validation_set = set(validation)
    train = [index for index in range(len(samples)) if index not in validation_set]
    if len(train) != len(samples) - validation_size or len(validation) != validation_size:
        raise AssertionError("split size invariant failed")
    if set(train) & validation_set:
        raise AssertionError("training and validation splits overlap")
    return train, sorted(validation)


class CompetitionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[LabeledImage],
        label_to_index: dict[str, int],
        transform: Callable | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.samples = samples
        self.label_to_index = label_to_index
        self.transform = transform
        self.indices = list(range(len(samples))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.samples[self.indices[item]]
        image = _open_rgb(sample.path)
        if image.size != (32, 32):
            raise ValueError(f"Expected 32x32 image, got {image.size}: {sample.path}")
        image = self.transform(image) if self.transform is not None else image
        return image, self.label_to_index[sample.label]


class UnlabeledCompetitionDataset(Dataset):
    def __init__(self, sample_submission: str | Path, image_dir: str | Path, transform=None,
                 test_metadata: str | Path | None = None):
        self.rows = read_csv(sample_submission)
        if list(self.rows[0]) != ['id', 'label']:
            raise ValueError('Sample submission must have exactly id,label columns')
        self.id_column, self.label_column = infer_columns(self.rows)
        self.ids = [row[self.id_column] for row in self.rows]
        if len(self.ids) != 2_000:
            raise ValueError(f"Expected 2,000 hidden-evaluation IDs, found {len(self.ids):,}")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate IDs in sample submission")
        if test_metadata is not None:
            metadata = read_csv(test_metadata)
            if set(metadata[0]) != {"id", "filename"}:
                raise ValueError("test metadata must have exactly id,filename columns")
            mapping = {row["id"]: row["filename"] for row in metadata}
            if len(mapping) != len(metadata) or set(mapping) != set(self.ids):
                raise ValueError("Test metadata IDs must match sample IDs one-to-one")
            if [row['id'] for row in metadata] != self.ids:
                raise ValueError('Test metadata IDs are reordered relative to sample submission')
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("Duplicate test filenames")
            self.paths = [resolve_image(Path(image_dir), mapping[i]) for i in self.ids]
        else:
            self.paths = [resolve_image(Path(image_dir), image_id) for image_id in self.ids]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image = _open_rgb(self.paths[index])
        if image.size != (32, 32):
            raise ValueError(f"Expected 32x32 image: {self.paths[index]}")
        image = self.transform(image) if self.transform is not None else image
        return image, self.ids[index]


def validate_training_distribution(samples: Sequence[LabeledImage]) -> dict[str, int]:
    counts = dict(sorted(Counter(sample.label for sample in samples).items()))
    if sum(counts.values()) != 50_000:
        raise ValueError("Only the 50,000 supplied training images may be used")
    if counts != {str(i): 5000 for i in range(10)}:
        raise ValueError("Expected 5,000 supplied images per integer class")
    return counts


def dataset_fingerprint(samples: Sequence[LabeledImage]) -> str:
    """Bind split indices to ordered filenames, labels AND training image bytes."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(f"{sample.image_id}\0{sample.path.name}\0{sample.label}\0".encode())
        digest.update(hashlib.sha256(sample.path.read_bytes()).digest())
    return digest.hexdigest()

