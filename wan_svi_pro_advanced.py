from typing_extensions import override
from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
from typing import Optional
import math


class WanSVIProAdvancedI2V(io.ComfyNode):
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanSVIProAdvancedI2V",
            display_name="Wan SVI Pro Advanced I2V",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                # 基础参数
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Width of the generated video in pixels"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Height of the generated video in pixels"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Total number of frames in the generated video"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number,
                           tooltip="Batch size (number of videos to generate)"),
                
                # 动态调整参数（最小值改为 1.0，表示无增强）
                io.Float.Input("motion_boost", default=1.0, min=1.0, max=3.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion amplitude amplification\n1.0 = no amplification (default)\n>1.0 = amplify movement"),
                io.Float.Input("detail_boost", default=1.0, min=1.0, max=4.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion dynamic strength\n1.0 = balanced (default)\n>1.0 = stronger motion dynamics"),
                io.Float.Input("motion_influence", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider,
                             tooltip="Influence strength of motion latent from previous video\n1.0 = normal, <1.0 = weaker, >1.0 = stronger"),
                
                # 重叠帧参数（统一使用，以图像帧为单位）
                io.Int.Input("overlap_frames", default=4, min=0, max=128, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Number of overlapping frames (pixel frames). Must be multiple of 4.\n4 pixel frames = 1 latent frame.\nControls how many frames from previous video to use as motion reference."),
                
                # 起始帧组
                io.Image.Input("start_image", optional=True,
                             tooltip="First frame reference image (anchor for the video)"),
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                               tooltip="Enable start frame conditioning"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in high-noise stage\n0.0 = no conditioning, 1.0 = full conditioning"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in low-noise stage"),
                
                # 中间帧组
                io.Image.Input("middle_image", optional=True,
                             tooltip="Middle frame reference image for better consistency"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                               tooltip="Enable middle frame conditioning"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Position of middle frame (0=start, 1=end)"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in high-noise stage"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in low-noise stage"),
                
                # 结束帧组
                io.Image.Input("end_image", optional=True,
                             tooltip="Last frame reference image (target ending)"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                               tooltip="Enable end frame conditioning"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for end frame in low-noise stage"),
                
                # 其他参数
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                        tooltip="CLIP vision embedding for start frame (for better semantic consistency)"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                        tooltip="CLIP vision embedding for middle frame"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                        tooltip="CLIP vision embedding for end frame"),
                
                # 原版核心输入（anchor_samples 保留）
                io.Latent.Input("anchor_samples", optional=True,
                              tooltip="Anchor latent samples (from VAE encode). Usually the first frame latent."),
                io.Latent.Input("prev_latent", optional=True,
                              tooltip="Previous video latent for seamless continuation"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number, 
                           optional=True, tooltip="Video frame offset (advanced, usually set to 0)\nSkip this many frames from input images"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="next_offset"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_boost=1.0, detail_boost=1.0, motion_influence=1.0,
                overlap_frames=4,
                start_image=None, enable_start_frame=True,
                high_noise_start_strength=1.0, low_noise_start_strength=1.0,
                middle_image=None, enable_middle_frame=True, middle_frame_ratio=0.5,
                high_noise_mid_strength=0.8, low_noise_mid_strength=0.2,
                end_image=None, enable_end_frame=True, low_noise_end_strength=1.0,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None,
                anchor_samples=None,              # 原版锚点潜变量
                prev_latent=None, video_frame_offset=0):
        
        # 重命名变量以保持代码一致性（仅用于增强模式）
        motion_amplification = motion_boost
        dynamic_strength = detail_boost
        
        # 计算基本参数
        spatial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1  # 1个潜变量帧 = 4个图像帧
        H = height // spatial_scale
        W = width // spatial_scale
        
        device = comfy.model_management.intermediate_device()
        
        # 创建空latent（后续采样使用）
        latent = torch.zeros([batch_size, latent_channels, total_latents, H, W], 
                            device=device)
        
        trim_latent = 0
        trim_image = 0
        next_offset = 0
        
        # 应用视频帧偏移（如果启用）
        if video_frame_offset > 0:
            if start_image is not None and start_image.shape[0] > 1:
                start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
            
            if middle_image is not None and middle_image.shape[0] > 1:
                middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
            
            if end_image is not None and end_image.shape[0] > 1:
                end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None
            
            next_offset = video_frame_offset + length
        
        # 计算中间位置
        middle_idx = cls._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        middle_latent_idx = middle_idx // 4
        
        # 调整图像尺寸
        def resize_image(img):
            if img is None:
                return None
            return comfy.utils.common_upscale(
                img[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        start_image = resize_image(start_image) if start_image is not None else None
        middle_image = resize_image(middle_image) if middle_image is not None else None
        end_image = resize_image(end_image) if end_image is not None else None
        
        # ========== 基础模式检测：完全复刻原版 WanImageToVideoSVIPro ==========
        is_basic_mode = (start_image is None and middle_image is None and end_image is None and
                         anchor_samples is not None and
                         motion_boost == 1.0 and detail_boost == 1.0 and motion_influence == 1.0)
        
        if is_basic_mode:
            print("[SVI Pro Advanced] Basic mode activated (replicating original WanImageToVideoSVIPro behavior).")
            
            # 提取 anchor latent（可能包含多帧）
            anchor_latent = anchor_samples["samples"].clone()  # shape: [B, C, T_anchor, H, W]
            
            # 从 overlap_frames 计算要取的潜变量帧数（图像帧数转潜变量帧数）
            motion_latent_frames_basic = max(0, overlap_frames // 4)  # 整除4，确保为整数
            
            # 处理 prev_latent（如果有）
            if prev_latent is not None and motion_latent_frames_basic > 0:
                prev_samples = prev_latent["samples"]
                use_frames = min(motion_latent_frames_basic, prev_samples.shape[2])
                if use_frames > 0:
                    motion_latent = prev_samples[:, :, -use_frames:].clone()
                    image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)
                else:
                    image_cond_latent = anchor_latent
            else:
                image_cond_latent = anchor_latent
            
            # 计算需要填充的帧数
            current_frames = image_cond_latent.shape[2]
            total_latents_needed = ((length - 1) // 4) + 1
            padding_size = total_latents_needed - current_frames
            
            # 填充到目标长度
            if padding_size > 0:
                padding = torch.zeros(1, latent_channels, padding_size, H, W,
                                    dtype=image_cond_latent.dtype, device=device)
                # 对填充部分应用 Wan21 的 process_out（与原版一致）
                padding = comfy.latent_formats.Wan21().process_out(padding)
                image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)
            else:
                # 如果提供的 latent 已经超过所需长度，则截断
                image_cond_latent = image_cond_latent[:, :, :total_latents_needed]
            
            # 创建掩码，形状为 [1, 1, total_latents, H, W]（与原版完全一致）
            mask = torch.ones((1, 1, total_latents_needed, H, W),
                              device=device, dtype=image_cond_latent.dtype)
            mask[:, :, :1] = 0.0  # 只有第一帧（anchor的第一帧）掩码为0，其余为1
            
            # 构建 conditioning
            positive_high = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            positive_low = positive_high  # 原版没有区分高低噪声，这里复用
            negative_out = node_helpers.conditioning_set_values(negative, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            
            # 创建空 latent（用于后续采样）
            out_latent = {"samples": latent}
            
            # 返回（注意返回顺序与输出定义一致）
            return io.NodeOutput(positive_high, positive_low, negative_out, out_latent,
                                 trim_latent, trim_image, next_offset)  # trim_* 均为0
        
        # ========== 如果不满足基础模式，继续执行原有的增强逻辑 ==========
        # 以下代码与原始增强逻辑完全相同，仅将 motion_latent_count 替换为 overlap_frames 转换后的值
        
        # 检查是否有prev_latent用于继续
        has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)
        
        if has_prev_latent:
            # SVI Continue: 使用prev_latent作为延续参考
            prev_samples = prev_latent["samples"]
            
            # 将像素帧转换为潜变量帧（基准帧数）
            motion_latent_frames = max(1, overlap_frames // 4)  # 至少1个潜变量帧
            
            # 根据动态强度调整使用的帧数
            adjusted_frames = min(motion_latent_frames, prev_samples.shape[2])
            
            # 动态强度影响使用的帧数
            if dynamic_strength > 1.0:
                use_frames = min(int(adjusted_frames * dynamic_strength), prev_samples.shape[2])
            else:
                use_frames = max(1, int(adjusted_frames * dynamic_strength))
            
            # 提取重叠潜变量帧
            motion_latent = prev_samples[:, :, -use_frames:].clone()
            
            # 核心改进：动作幅度放大（仅在 motion_amplification != 1.0 时生效）
            if use_frames >= 2 and motion_amplification != 1.0:
                # 计算运动向量（帧间差异）
                motion_vectors = []
                for i in range(1, use_frames):
                    vector = motion_latent[:, :, i] - motion_latent[:, :, i-1]
                    motion_vectors.append(vector)
                
                if motion_vectors:
                    amplified_vectors = [vec * motion_amplification for vec in motion_vectors]
                    amplified_latent = [motion_latent[:, :, 0:1].clone()]
                    
                    for i in range(len(amplified_vectors)):
                        next_frame = amplified_latent[-1] + amplified_vectors[i].unsqueeze(2)
                        amplified_latent.append(next_frame)
                    
                    motion_latent = torch.cat(amplified_latent, dim=2)
                    print(f"[SVI Pro] Motion amplification applied: {motion_amplification:.1f}x")
            
            # 应用运动强度（motion_influence）
            if motion_influence != 1.0:
                motion_latent = motion_latent * motion_influence
            
            # 根据分辨率自动提示（不影响逻辑）
            resolution_factor = math.sqrt(width * height) / math.sqrt(832 * 480)
            if resolution_factor > 1.2:
                print(f"[SVI Pro] High resolution detected ({width}x{height}), consider using dynamic_strength > 1.5 for better motion")
            
            # 构建统一的image_cond_latent
            if start_image is not None:
                anchor_latent = vae.encode(start_image[:1, :, :, :3])
            else:
                anchor_latent = torch.zeros([1, latent_channels, 1, H, W], 
                                           device=device, dtype=latent.dtype)
            
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           dtype=anchor_latent.dtype, device=anchor_latent.device)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            if enable_start_frame:
                image_cond_latent[:, :, :1] = anchor_latent
            
            # 将运动潜变量放入合适位置
            motion_start = 1 if enable_start_frame else 0
            motion_end = min(motion_start + use_frames, total_latents)
            
            if motion_end > motion_start:
                motion_to_use = motion_latent[:, :, :motion_end-motion_start]
                image_cond_latent[:, :, motion_start:motion_end] = motion_to_use
            
            # 插入中间图像
            if middle_image is not None and enable_middle_frame:
                middle_latent = vae.encode(middle_image[:1, :, :, :3])
                if middle_latent_idx < total_latents:
                    actual_middle_idx = middle_latent_idx
                    while (actual_middle_idx < motion_end and actual_middle_idx < total_latents):
                        actual_middle_idx += 1
                    if actual_middle_idx < total_latents:
                        image_cond_latent[:, :, actual_middle_idx:actual_middle_idx+1] = middle_latent
                        middle_latent_idx = actual_middle_idx
            
            # 插入结束图像
            if end_image is not None and enable_end_frame:
                end_latent = vae.encode(end_image[:1, :, :, :3])
                image_cond_latent[:, :, total_latents-1:total_latents] = end_latent
            
            # 根据dynamic_strength计算衰减率
            if dynamic_strength <= 1.0:
                decay_rate = 0.9 - (dynamic_strength - 0.5) * 0.4
            else:
                decay_rate = 0.7 - (dynamic_strength - 1.0) * 0.2
            decay_rate = max(0.05, min(0.9, decay_rate))
            
            # 创建无缝衔接掩码（4通道）
            mask_high = torch.ones((1, 4, total_latents, H, W), 
                                  device=device, dtype=anchor_latent.dtype)
            mask_low = torch.ones((1, 4, total_latents, H, W), 
                                 device=device, dtype=anchor_latent.dtype)
            
            if enable_start_frame and start_image is not None:
                mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
            
            if motion_end > motion_start:
                for i in range(motion_start, motion_end):
                    distance = i - motion_start
                    decay = decay_rate ** distance
                    mask_high_val = 1.0 - (high_noise_start_strength * decay)
                    mask_high[:, :, i:i+1] = max(0.05, min(0.95, mask_high_val))
                    mask_low_val = 1.0 - (low_noise_start_strength * decay * 0.7)
                    mask_low[:, :, i:i+1] = max(0.1, min(0.95, mask_low_val))
            
            if middle_image is not None and enable_middle_frame and middle_latent_idx < total_latents:
                mask_high[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None and enable_end_frame:
                mask_high[:, :, total_latents-1:total_latents] = 0.0
                mask_low[:, :, total_latents-1:total_latents] = max(0.0, 1.0 - low_noise_end_strength)
            
            # 构建条件
            positive_high_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_high
            })
            
            positive_low_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_low
            })
            
            negative_out = node_helpers.conditioning_set_values(negative, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_high
            })
            
            # 处理clip vision
            clip_vision_output = cls._merge_clip_vision_outputs(
                clip_vision_start_image if enable_start_frame else None, 
                clip_vision_middle_image if enable_middle_frame else None, 
                clip_vision_end_image if enable_end_frame else None
            )
            
            if clip_vision_output is not None:
                positive_low_noise = node_helpers.conditioning_set_values(
                    positive_low_noise, 
                    {"clip_vision_output": clip_vision_output}
                )
                negative_out = node_helpers.conditioning_set_values(
                    negative_out,
                    {"clip_vision_output": clip_vision_output}
                )
            
            out_latent = {"samples": latent}
            
            return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                    trim_latent, trim_image, next_offset)
        
        elif start_image is not None:
            # 如果没有prev_latent，仅使用起始图像（与原有逻辑相同）
            anchor_latent = vae.encode(start_image[:1, :, :, :3])
            
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           dtype=anchor_latent.dtype, device=anchor_latent.device)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            if enable_start_frame:
                image_cond_latent[:, :, :1] = anchor_latent
            
            if middle_image is not None and enable_middle_frame:
                middle_latent = vae.encode(middle_image[:1, :, :, :3])
                if middle_latent_idx < total_latents:
                    image_cond_latent[:, :, middle_latent_idx:middle_latent_idx+1] = middle_latent
            
            if end_image is not None and enable_end_frame:
                end_latent = vae.encode(end_image[:1, :, :, :3])
                image_cond_latent[:, :, total_latents-1:total_latents] = end_latent
            
            mask_high = torch.ones((1, 4, total_latents, H, W), 
                                  device=device, dtype=anchor_latent.dtype)
            mask_low = torch.ones((1, 4, total_latents, H, W), 
                                 device=device, dtype=anchor_latent.dtype)
            
            if enable_start_frame:
                mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
            
            if middle_image is not None and enable_middle_frame:
                mask_high[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None and enable_end_frame:
                mask_high[:, :, total_latents-1:total_latents] = 0.0
                mask_low[:, :, total_latents-1:total_latents] = max(0.0, 1.0 - low_noise_end_strength)
            
            positive_high_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_high
            })
            
            positive_low_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_low
            })
            
            negative_out = node_helpers.conditioning_set_values(negative, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask_high
            })
            
            clip_vision_output = cls._merge_clip_vision_outputs(
                clip_vision_start_image if enable_start_frame else None, 
                clip_vision_middle_image if enable_middle_frame else None, 
                clip_vision_end_image if enable_end_frame else None
            )
            
            if clip_vision_output is not None:
                positive_low_noise = node_helpers.conditioning_set_values(
                    positive_low_noise, 
                    {"clip_vision_output": clip_vision_output}
                )
                negative_out = node_helpers.conditioning_set_values(
                    negative_out,
                    {"clip_vision_output": clip_vision_output}
                )
            
            out_latent = {"samples": latent}
            
            return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                    trim_latent, trim_image, next_offset)
        else:
            # 没有任何起始信息：创建基本的空条件（回退）
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           device=device, dtype=latent.dtype)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            mask = torch.ones((1, 4, total_latents, H, W), 
                            device=device, dtype=latent.dtype)
            
            positive_high_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            
            negative_out = node_helpers.conditioning_set_values(negative, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            
            out_latent = {"samples": latent}
            
            return io.NodeOutput(positive_high_noise, positive_high_noise, negative_out, out_latent,
                    trim_latent, trim_image, next_offset)
    
    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx
    
    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
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


# ===========================================
# 节点注册
# ===========================================
NODE_CLASS_MAPPINGS = {
    "WanSVIProAdvancedI2V": WanSVIProAdvancedI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIProAdvancedI2V": "Wan SVI Pro Advanced I2V",
}
