import os


def get_image_list(image_path, valid_suffix=None, filter_key=None):
    """Get image list from image name or image directory name with valid suffix.
    if needed, filter_key can be used to whether 'include' the key word.
    When filter_key is not None. it indicates whether filenames should include certain key.

    Args:
    image_path(str): the image or image folder where you want to get a image list from.
    valid_suffix(tuple): Contain only the suffix you want to include.
    filter_key(dict): the key(ignore case) and whether you want to include it. e.g.:{"segmentation": True} will further filter the imagename with segmentation in it.

    """
    if valid_suffix is None:
        valid_suffix = [
            'nii.gz', 'nii', 'dcm', 'nrrd', 'mhd', 'raw', 'npy', 'mha'
        ]

    image_list = []
    if os.path.isfile(image_path):
        if image_path.split("/")[-1].split(
                '.', maxsplit=1)[-1] in valid_suffix:
            if filter_key is not None:
                f_name = image_path.split("/")[
                    -1]  # TODO change to system invariant
                for key, val in filter_key.items():
                    if (key in f_name.lower()) is not val:
                        break
                else:
                    image_list.append(image_path)
            else:
                image_list.append(image_path)
        else:
            raise FileNotFoundError(
                '{} is not a file end with supported suffix, the support suffixes are {}.'
                .format(image_path, valid_suffix))

    # load image in a directory
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.split(".", maxsplit=1)[-1] in valid_suffix:
                    if filter_key is not None:
                        for key, val in filter_key.items():
                            if (key in f.lower()) is not val:
                                break
                        else:
                            image_list.append(os.path.join(root, f))
                    else:
                        image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '{} is not found. it should be a path of image, or a directory including images.'
            .format(image_path))

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list
