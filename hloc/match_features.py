import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union
import os

# Avoid HDF5 file locking to prevent "file is already open" errors
# from crashed runs or concurrent reads
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import torch
import torch.distributed as dist
from joblib import Parallel, delayed
from tqdm import tqdm

from . import logger, matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "superpoint+lightglue": {
        "output": "matches-superpoint-lightglue",
        "model": {
            "name": "lightglue",
            "features": "superpoint",
        },
    },
    "disk+lightglue": {
        "output": "matches-disk-lightglue",
        "model": {
            "name": "lightglue",
            "features": "disk",
        },
    },
    "aliked+lightglue": {
        "output": "matches-aliked-lightglue",
        "model": {
            "name": "lightglue",
            "features": "aliked",
        },
    },
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
        },
    },
    "superglue-fast": {
        "output": "matches-superglue-it5",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
        },
    },
    "adalam": {
        "output": "matches-adalam",
        "model": {"name": "adalam"},
    },
}


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size
    return 0, 1


def check_pairs(pairs_chunk, existing_keys):
    res = []
    for i, j in pairs_chunk:
        if (
            names_to_pair(i, j) in existing_keys
            or names_to_pair(j, i) in existing_keys
            or names_to_pair_old(i, j) in existing_keys
            or names_to_pair_old(j, i) in existing_keys
        ):
            continue
        res.append((i, j))
    return res


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    # Use "r+" which is less aggressive than "a" regarding locks, or retry logic could be added
    # But disabling locking via env var is the most effective fix for this specific error.
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        try:
            with h5py.File(str(match_path), "r", libver="latest") as fd:
                existing_keys = set(fd.keys())
        except OSError:
            logger.warning(
                f"Could not open {match_path} (likely locked/corrupted). "
                "Deleting and restarting."
            )
            match_path.unlink()
            existing_keys = set()

        # Parallel filtering using joblib
        chunk_size = 10_000
        if len(pairs) > chunk_size:
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(check_pairs)(pairs[i : i + chunk_size], existing_keys)
                for i in range(0, len(pairs), chunk_size)
            )
            pairs_filtered = [p for res in results for p in res]
            return pairs_filtered

        pairs_filtered = []
        for i, j in pairs:
            if (
                names_to_pair(i, j) in existing_keys
                or names_to_pair(j, i) in existing_keys
                or names_to_pair_old(i, j) in existing_keys
                or names_to_pair_old(j, i) in existing_keys
            ):
                continue
            pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    rank, world_size = init_distributed()
    if world_size > 1:
        match_path = match_path.parent / (
            match_path.stem + f"_rank{rank}" + match_path.suffix
        )

    if rank == 0:
        logger.info(
            "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
        )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]

    if world_size > 1:
        pairs = pairs[rank::world_size]

    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info(f"Rank {rank}: Skipping the matching.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=5,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        # pre_fetch=16,
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1, disable=rank != 0, desc=f"Rank {rank}")):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info(f"Rank {rank}: Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-superpoint-n4096-r1024")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
