
from pathlib import Path
import argparse
import re
import shutil
import tarfile
import zipfile

import numpy as np
import torch

from data.imagenet_dataset import ImageNetDataset
from model import create_model


def extract_one_npz(archive, destination):
    """Extract one genuine event sample without extracting the whole archive."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()

        for name in names:
            if name.lower().endswith(".npz"):
                with z.open(name) as source:
                    with destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                return name

        for name in names:
            if name.lower().endswith((".tar.gz", ".tgz", ".tar")):
                with z.open(name) as stream:
                    mode = "r|gz" if name.lower().endswith((".tar.gz", ".tgz")) else "r|"
                    with tarfile.open(fileobj=stream, mode=mode) as tar:
                        for member in tar:
                            if member.isfile() and member.name.lower().endswith(".npz"):
                                with tar.extractfile(member) as source:
                                    with destination.open("wb") as target:
                                        shutil.copyfileobj(source, target)
                                return f"{name} :: {member.name}"

    raise RuntimeError(
        "No NPZ file found in the archive. Inspect its structure before proceeding."
    )


def main(archive):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_path = Path("/kaggle/working/sample_event.npz")

    print("STEP 1: Extracting a real sample")
    original_name = extract_one_npz(archive, sample_path)
    print("Source:", original_name)

    print("\nSTEP 2: Inspecting NPZ")
    with np.load(sample_path, allow_pickle=False) as data:
        print("Keys:", data.files)
        compressed = "event_data" in data.files

    print("\nSTEP 3: Original STP preprocessing")
    processor = ImageNetDataset.__new__(ImageNetDataset)

    cfg = {
        "compressed": compressed,
        "reshape": True,
        "reshape_method": "no_sample",
        "slice_events": True,
        "slice_method": "time",
        "slice_start": -1,
        "slice_end": 30000,
    }

    events = processor.parse_event(str(sample_path), cfg)

    if len(events) == 0:
        raise RuntimeError("No events after original time slicing.")

    print("Events shape:", tuple(events.shape))
    print("Number of events:", len(events))

    tecm = processor.reshape_then_acc_count_pol(
        events.clone(), "eval"
    )

    print("TECM shape:", tuple(tecm.shape))
    print("TECM total counts:", int(tecm.sum().item()))

    assert tecm.shape == (5, 2, 224, 224)
    assert torch.isfinite(tecm).all()
    assert tecm.sum().item() > 0

    print("\nSTEP 4: Creating STP model")
    model = create_model("STP_ViT_S")
    model = model.to(device)
    model.eval()

    x = tecm.unsqueeze(0).to(device)
    print("Model input:", tuple(x.shape))

    print("\nSTEP 5: Forward Pass")
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=5, dim=1)

    print("Logits shape:", tuple(logits.shape))
    print("Top-5 class indices:", indices[0].cpu().tolist())
    print("Top-5 probabilities:", values[0].cpu().tolist())

    assert logits.shape == (1, 1000)
    assert torch.isfinite(logits).all()

    synsets = re.findall(r"n\d{8}", original_name)
    print("Sample synset:", synsets[-1] if synsets else "Not identified")
    print("\nPASS: REAL EVENT DATA -> STP -> CLASSIFICATION LOGITS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    main(args.archive)
