import pathlib
import numpy as np
from xaosim.shmlib import shm
import argparse
import os
import glob

base_pth = pathlib.Path("~/.config/DM_flats/")
categories = [
    "lab",
    "night-standard",
    "night-faint",
    "factory",
    "test",
]

flat_pth = pathlib.Path("/usr/local/etc/DMShapes/")
factory_flat_cmd_files = {
    1: "17DW019#113_FLAT_MAP_COMMANDS.txt",
    2: "17DW019#053_FLAT_MAP_COMMANDS.txt",
    3: "17DW019#093_FLAT_MAP_COMMANDS.txt",
    4: "17DW019#122_FLAT_MAP_COMMANDS.txt",
}

# files are saved in base_pth/beam{beam_id}/{category}_{timestamp}.npy
file_naming_format = "{category}_{timestamp}.npy"
valid_beam_ids = [1, 2, 3, 4]


def load_factory_flat(beam_no):
    # the .txt flat is just a list with each index on a newline
    flat_file = factory_flat_cmd_files[beam_no]
    return load_from_txt(flat_pth / flat_file)


def load_from_txt(file_name):
    data = np.loadtxt(file_name)
    data = np.insert(data, [0, 10, 130, 140], 0.0).reshape((12, 12))
    return data


def load_from_npy(file_name):
    data = np.load(file_name)
    return data


def load_flat(beam_no, category):
    # construct the file path from the category and beam id
    beam_pth = base_pth.expanduser() / f"beam{beam_no}"
    # find the most recent file in the category
    category_files = glob.glob(str(beam_pth / f"{category}_*.npy"))
    print(str(beam_pth / f"{category}_*.npy"))
    print(category_files)
    if not category_files:
        print(f"No files found for category {category} and beam {beam_no}.")
        return None
    # if more than one file, raise a warning and take the most recent one
    if len(category_files) > 1:
        print(
            f"Warning: multiple files found for category {category} and beam {beam_no}. Loading the most recent one."
        )
    file_pth = max(category_files, key=os.path.getctime)

    return load_from_npy(file_pth)


def command_dm(beam_no, data):
    s = shm(f"/dev/shm/dm{beam_no}disp00.im.shm", nosem=False)
    print("writing", data.shape)

    s.set_data(data)
    s0 = shm(f"/dev/shm/dm{beam_no}.im.shm", nosem=False)
    s0.post_sems()
    return True


def main():
    parser = argparse.ArgumentParser(description="Load a factory flat file to the DM.")
    parser.add_argument(
        "beam_id",
        type=int,
        choices=[-1, *valid_beam_ids],
        help="The beam ID of the DM to load (1-4), or -1 to load all beams.",
    )
    parser.add_argument(
        "category",
        type=str,
        choices=categories,
        help="The category of the flat to load.",
    )
    # alternatively, the user can specify a file path to load
    parser.add_argument(
        "--file",
        type=pathlib.Path,
        help="The path to the flat file to load. Overrides category and beam if specified.",
    )
    args = parser.parse_args()

    beam_ids_to_load = valid_beam_ids if args.beam_id == -1 else [args.beam_id]

    for beam_id in beam_ids_to_load:
        if args.file is not None:
            file_pth = args.file
            if file_pth.suffix == ".npy":
                data = load_from_npy(file_pth)
            else:
                data = load_from_txt(file_pth)
        elif args.category == "factory":
            data = load_factory_flat(beam_id)
            file_pth = "factory"
        else:
            # construct the file path from the category and beam id
            beam_pth = base_pth.expanduser() / f"beam{beam_id}"
            # find the most recent file in the category
            category_files = glob.glob(str(beam_pth / f"{args.category}_*.npy"))
            if not category_files:
                print(
                    f"No files found for category {args.category} and beam {beam_id}."
                )
                continue
            # if more than one file, raise a warning and take the most recent one
            if len(category_files) > 1:
                print(
                    f"Warning: multiple files found for category {args.category} "
                    f"and beam {beam_id}. Loading the most recent one."
                )
            file_pth = max(category_files, key=os.path.getctime)
            print(file_pth)
            data = load_from_npy(file_pth)

        s = command_dm(beam_id, data)
        if s:
            print(f"Loaded flat file {file_pth} to DM{beam_id}.")
