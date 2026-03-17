import torch
import torch.nn.functional as F
import comfy.clip_vision


def merge_clip_vision_outputs(*outputs):
    """Merge multiple CLIP vision outputs by concatenating hidden states."""
    valid_outputs = [o for o in outputs if o is not None]
    if not valid_outputs:
        return None
    if len(valid_outputs) == 1:
        return valid_outputs[0]
    all_states = [o.penultimate_hidden_states for o in valid_outputs]
    combined_states = torch.cat(all_states, dim=-2)
    result = comfy.clip_vision.Output()
    result.penultimate_hidden_states = combined_states
    return result


def create_spatial_gradient(img1, img2, mask_h, mask_w, boost_factor):
    """Compute a spatial gradient mask from the L1 motion diff between two images.

    Returns a (mask_h, mask_w) tensor, or None if either input is None.
    """
    if img1 is None or img2 is None:
        return None

    motion_diff = torch.abs(img2[0] - img1[0]).mean(dim=-1, keepdim=False)
    motion_diff_4d = motion_diff.unsqueeze(0).unsqueeze(0)
    motion_diff_scaled = F.interpolate(
        motion_diff_4d,
        size=(mask_h, mask_w),
        mode='bilinear',
        align_corners=False
    )

    motion_normalized = (motion_diff_scaled - motion_diff_scaled.min()) / (motion_diff_scaled.max() - motion_diff_scaled.min() + 1e-8)

    spatial_gradient = 1.0 - motion_normalized * boost_factor * 2.5
    spatial_gradient = torch.clamp(spatial_gradient, 0.02, 1.0)
    return spatial_gradient[0, 0]
