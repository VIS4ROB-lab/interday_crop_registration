import io
from loguru import logger

import cv2
import numpy as np
import h5py
import torch
from numpy.linalg import inv


try:
    # for internel use only
    from .client import MEGADEPTH_CLIENT, SCANNET_CLIENT
except Exception:
    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_array_from_s3(
    path, client, cv_type,
    use_h5py=False,
):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, client=SCANNET_CLIENT):
    cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
    if str(path).startswith('s3://'):
        image = load_array_from_s3(str(path), client, cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_cropped_image(image, w_cropped):
    w, h = image.shape[1], image.shape[0]
    scale = w_cropped / w
    w_new = w_cropped
    h_new = int(scale * h)
    w_start = int((w - w_new) / 2)
    w_end = w_start + w_new
    h_start = int((h - h_new) / 2)
    h_end = h_start + h_new

    return image[h_start:h_end, w_start:w_end]


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask

def pad_bottom_right_cut(inp, pad_size):
    assert isinstance(pad_size, int) and inp.ndim == 2, "height map should has and only has 2 dims"
    shape_0_end = min(inp.shape[0], pad_size)
    shape_1_end = min(inp.shape[1], pad_size)
    padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
    padded[:shape_0_end, :shape_1_end] = inp[:shape_0_end, :shape_1_end]
    return padded

def set_boarder_to_zero(inp):
    inp[0, :] = 0
    inp[-1, :] = 0
    inp[:, 0] = 0
    inp[:, -1] = 0
    return inp

def cut_square_from_heightmap(height_map, center_x, center_y, side_length):
    """
    Cut a square from a PyTorch Tensor based on center coordinates and side length.

    Args:
        input_tensor (torch.Tensor): Input n*n PyTorch Tensor.
        center_x (int): X-coordinate of the center of the new square.
        center_y (int): Y-coordinate of the center of the new square.
        side_length (int): Length of the side of the square.

    Returns:
        torch.Tensor: A new square Tensor cut from the input Tensor, padded with zeros.
    """
    n = height_map.shape[0]  # Assuming input_tensor is n*n

    # Calculate the coordinates for cropping the square
    start_x = max(center_x - side_length // 2, 0)
    end_x = min(center_x + side_length // 2, n)
    start_y = max(center_y - side_length // 2, 0)
    end_y = min(center_y + side_length // 2, n)

    # Create an empty square tensor filled with zeros
    square_tensor = torch.zeros(side_length, side_length, dtype=height_map.dtype)

    # Calculate the corresponding region in the input tensor
    input_region = height_map[start_y:end_y, start_x:end_x]

    # Calculate the size of the region in both dimensions
    region_height, region_width = input_region.shape

    # Calculate the starting positions for copying into the square tensor
    copy_start_x = max(side_length // 2 - center_x, 0)
    copy_start_y = max(side_length // 2 - center_y, 0)

    # Calculate the ending positions for copying into the square tensor
    copy_end_x = copy_start_x + region_width
    copy_end_y = copy_start_y + region_height

    # Copy the input region into the square tensor
    square_tensor[copy_start_y:copy_end_y, copy_start_x:copy_end_x] = input_region

    start_x = center_x - side_length // 2
    start_y = center_y - side_length // 2

    return square_tensor, start_x, start_y

# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)

    # resize image
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask)

    return image, mask, scale


def read_megadepth_depth(path, pad_to=None):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


def read_scannet_depth(path):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]


# --- CROP ---

def read_crop_gray(path, resize=None, df=None, augment_fn=None, crop=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects.
        crop (optional): the longer edge of the cropped images. Crop the image after read, before resize.
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn, client=MEGADEPTH_CLIENT)
    w, h = image.shape[1], image.shape[0]
    # print('ori image', image.shape)

    # crop image
    if crop:
        image = get_cropped_image(image, crop)
        w, h = image.shape[1], image.shape[0]
        # print('after crop', image.shape)

    # resize image
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    # print('after image', image.shape)
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized

    return image, scale


def read_crop_depth(path, crop=None):
    if str(path).startswith('s3://'):
        depth = load_array_from_s3(path, MEGADEPTH_CLIENT, None, use_h5py=True)
    else:
        depth = np.array(h5py.File(path, 'r')['depth_data']).squeeze()

    # crop image
    if crop:
        depth = get_cropped_image(depth, crop)
        # print('depth after crop', depth.shape)

    depth = torch.from_numpy(depth).float()  # (h, w)
    # print('depth.shape', depth.shape)
    return depth


def read_crop_height_map(height_map_paths, pad_size):
    """
    read height map of the interested field area from file
    pad or cut size to (pad_size, pad_size)
    """

    height_maps = {}
    for date, path in height_map_paths.items():
        height_map_dict = dict(np.load(path, allow_pickle=True))
        height_map_info = np.array([height_map_dict['cell_size'],
                                    height_map_dict['x_min'],
                                    height_map_dict['y_min'],
                                    height_map_dict['x_max'],
                                    height_map_dict['y_max']
        ])
        height_map_info = torch.from_numpy(height_map_info)

        height_map = height_map_dict['height_map']
        height_map = pad_bottom_right_cut(height_map, pad_size)
        height_map = set_boarder_to_zero(height_map)

        height_map = torch.from_numpy(height_map).float()  # (6000, 6000)
        # print('height_map.shape', height_map.shape)

        height_maps[date] = (height_map, height_map_info)

    return height_maps

def cut_crop_height_map(height_map_tuple, pose, cut_size):
    """
    select only part of the raw height map that covers the area of the image
    to shrink the size while keep high resolution
    output size should be (cut_size, cut_size)
    """
    height_map, height_map_info = height_map_tuple
    cell_size = height_map_info[0].item()
    x_min = height_map_info[1].item()
    y_min = height_map_info[2].item()
    
    original_size = height_map.shape[0]
    assert original_size >= cut_size, f"original size smaller than cut size{original_size} < {cut_size}"

    x0 = round((pose[0,3].item()-x_min)/cell_size)
    y0 = round((pose[1,3].item()-y_min)/cell_size)

    height_map_new, x_start, y_start = cut_square_from_heightmap(height_map, x0, y0, cut_size)
    height_map_new = set_boarder_to_zero(height_map_new)

    x_min_new = x_min + cell_size * x_start
    x_max_new = x_min_new + cell_size * cut_size

    y_min_new = y_min + cell_size * y_start
    y_max_new = y_min_new + cell_size * cut_size

    height_map_info_new = np.array([cell_size,
                                    x_min_new,
                                    y_min_new,
                                    x_max_new,
                                    y_max_new
                                    ])
    height_map_info_new = torch.from_numpy(height_map_info_new)

    return (height_map_new, height_map_info_new)
