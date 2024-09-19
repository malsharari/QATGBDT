import numpy as np

def update_npz_file(filename, **arrays):
    """
    Update or create a .npz file with the given arrays.

    :param filename: Name of the .npz file to update or create.
    :param arrays: Arrays to save, passed as keyword arguments.
    """
    try:
        # Load existing data if file exists
        existing_data = np.load(filename, allow_pickle=True)
        existing_data = dict(existing_data)
    except FileNotFoundError:
        existing_data = {}

    # Update with new arrays
    existing_data.update(arrays)

    # Save all arrays to the .npz file
    np.savez(filename, **existing_data)