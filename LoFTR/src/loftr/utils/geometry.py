import torch


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 4, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    torch.set_printoptions(profile="full")
    # print("kpts0", kpts0_long)
    # print("-----------------------------")

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # print("kpts0_depth", kpts0_depth)
    # print("-----------------------------")

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # print("w_kpts0", w_kpts0)
    # print("-----------------------------")

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    # print("nonzero_mask", torch.sum(nonzero_mask), nonzero_mask.shape)
    # print("covisible_mask", torch.sum(covisible_mask), covisible_mask.shape)
    # print("consistent_mask", torch.sum(consistent_mask), consistent_mask.shape)
    # print("valid_mask", torch.sum(valid_mask), valid_mask.shape)

    return valid_mask, w_kpts0

@torch.no_grad()
def warp_kpts_chd(kpts0, depth0, depth1, height_map0, height_map_info0, T0, T1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Compensate for Height Difference using height maps of real crop images.
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        height_map0 (torch.Tensor): [N, H, W],
        height_map_info0 (torch.Tensor): [N, 5], [cell_size, x_min, y_min, x_max, y_max], 
        T0 (torch.Tensor): [N, 4, 4],
        T1 (torch.Tensor): [N, 4, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """

    cell_size = height_map_info0[:,0].reshape(-1,1,1)
    x_min = height_map_info0[:,1].reshape(-1,1,1)
    y_min = height_map_info0[:,2].reshape(-1,1,1)
    xy_min = torch.cat([x_min, y_min], dim=-1)

    kpts0_long = kpts0.round().long()

    torch.set_printoptions(profile="full")
    # print("kpts0", kpts0_long)
    # print("-----------------------------")

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # To Ground
    w_kpts0_ground = T0[:, :3, :3] @ kpts0_cam + T0[:, :3, [3]]    # (N, 3, L)

    # Get x, y coordinates
    w_kpts0_ground_xy = w_kpts0_ground[:, :2, :].transpose(1, 2)    # (N, L, 2)
    
    # Move indices to the range of the height map
    kpts0_height_map_indices = (
        (w_kpts0_ground_xy - xy_min) / cell_size
    ).to(torch.int64)
    
    # Clip indices to stay within valid range
    num_y, num_x = height_map0.shape[1], height_map0.shape[2]
    clipped_x_indices = torch.clamp(kpts0_height_map_indices[:, :, 0], 0, num_x - 1)
    clipped_y_indices = torch.clamp(kpts0_height_map_indices[:, :, 1], 0, num_y - 1)
    
    # Query height map to get the new z
    batch_indices = torch.arange(kpts0.shape[0], device=kpts0.device)
    kpts0_height_map = height_map0[
        batch_indices[:,None],
        clipped_x_indices,
        clipped_y_indices
    ]  # (N, L)
    height_nonzero_mask = kpts0_height_map != 0
    nonzero_mask *= height_nonzero_mask

    # replace z with the new ones
    w_kpts0_ground[:, 2, :] = kpts0_height_map

    # To Cam1
    T1_inv = T1.inverse()
    w_kpts0_cam = T1_inv[:, :3, :3] @ w_kpts0_ground + T1_inv[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    w_kpts0[~nonzero_mask] = 0  # if height is invalid, or depth is 0, warp the point to the left-up corner

    # print("w_kpts0", w_kpts0)
    # print("-----------------------------")

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    # print("nonzero_mask", torch.sum(nonzero_mask), nonzero_mask.shape)
    # print("covisible_mask", torch.sum(covisible_mask), covisible_mask.shape)
    # print("consistent_mask", torch.sum(consistent_mask), consistent_mask.shape)
    # print("valid_mask", torch.sum(valid_mask), valid_mask.shape)

    return valid_mask, w_kpts0