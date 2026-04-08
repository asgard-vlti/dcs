import pathlib
import numpy as np
from xaosim.shmlib import shm
import argparse
import datetime
import glob

base_pth = pathlib.Path("~/.config/DM_flats/")

categories = [
    "lab",
    "night-standard",
    "night-faint",
]

file_naming_format = "{category}_{timestamp}.npy"


def read_dm_total(beam_idx):
    s = shm(f"/dev/shm/dm{beam_idx}.im.shm", nosem=False)
    return s.get_data()


def push_to_archive(pth, file):
    # move the file from path/file to path/archive/file
    archive_pth = pth / "archive"
    archive_pth.mkdir(parents=True, exist_ok=True)
    file.rename(archive_pth / file.name)


def main():
    parser = argparse.ArgumentParser(
        description="Save the current DM shape to a flat file."
    )
    parser.add_argument(
        "beam_id",
        type=int,
        choices=[1, 2, 3, 4],
        help="The beam ID of the DM to save (1-4).",
    )
    parser.add_argument(
        "category",
        type=str,
        choices=categories,
        help="The category to save the flat under.",
    )
    args = parser.parse_args()

    # Get the current DM shape
    dm_shape = read_dm_total(args.beam_id)

    # Create the directory for the beam if it doesn't exist
    beam_pth = base_pth.expanduser() / f"beam{args.beam_id}"
    beam_pth.mkdir(parents=True, exist_ok=True)

    # Create the filename with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = file_naming_format.format(category=args.category, timestamp=timestamp)
    file_pth = beam_pth / filename

    # Save the DM shape to a .npy file
    np.save(file_pth, dm_shape)
    print(f"Saved DM shape to {file_pth}")

    # move the older file to the archive if it exists
    existing_files = glob.glob(str(beam_pth / f"{args.category}_*.npy"))
    for existing_file in existing_files:
        existing_file_pth = pathlib.Path(existing_file)
        if existing_file_pth != file_pth:
            push_to_archive(beam_pth, existing_file_pth)
            print(f"Moved old file {existing_file_pth} to archive.")
