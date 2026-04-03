# Wan I2V Modular Nodes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `wan_advanced_i2v.py` into a lean base node plus chainable option nodes, so users only connect the settings they need.

**Architecture:** A new custom type `WAN_I2V_OPTIONS` (a plain Python dict) flows through a chain of option nodes, each merging its settings in. The base node unpacks this dict (falling back to defaults for missing keys) and runs the generation logic. All new nodes live in `wan_i2v_modular.py`. `wan_advanced_i2v.py` is untouched.

**Tech Stack:** PyTorch, ComfyUI `comfy_api.latest`, `node_helpers`, existing `utils.py`

---

## Reference: WAN_I2V_OPTIONS dict schema

All keys are optional with these defaults:

```python
{
    # Strengths
    "high_noise_start_strength": 1.0,
    "high_noise_mid_strength":   0.8,
    "low_noise_start_strength":  1.0,
    "low_noise_mid_strength":    0.2,
    "low_noise_end_strength":    1.0,
    # Continuation
    "long_video_mode":       "DISABLED",   # DISABLED | AUTO_CONTINUE | SVI | LATENT_CONTINUE
    "motion_frames":         None,
    "prev_latent":           None,
    "continue_frames_count": 5,
    "svi_motion_strength":   1.0,
    "video_frame_offset":    0,
    # Clip vision
    "clip_vision_start_image":  None,
    "clip_vision_middle_image": None,
    "clip_vision_end_image":    None,
}
```

---

## Task 1: Create `wan_i2v_modular.py` skeleton

**Files:**
- Create: `wan_i2v_modular.py`

**Step 1: Write the file with imports and the OPTIONS_DEFAULTS constant only**

```python
from comfy_api.latest import io
import torch
import torch.nn.functional as F
import node_helpers
import comfy
import comfy.utils
import comfy.latent_formats
from .utils import merge_clip_vision_outputs, create_spatial_gradient

OPTIONS_DEFAULTS = {
    "high_noise_start_strength": 1.0,
    "high_noise_mid_strength":   0.8,
    "low_noise_start_strength":  1.0,
    "low_noise_mid_strength":    0.2,
    "low_noise_end_strength":    1.0,
    "long_video_mode":           "DISABLED",
    "motion_frames":             None,
    "prev_latent":               None,
    "continue_frames_count":     5,
    "svi_motion_strength":       1.0,
    "video_frame_offset":        0,
    "clip_vision_start_image":   None,
    "clip_vision_middle_image":  None,
    "clip_vision_end_image":     None,
}
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('wan_i2v_modular.py').read()); print('OK')"
```

---

## Task 2: Implement `WanStrengthOptions`

**Files:**
- Modify: `wan_i2v_modular.py`

**Step 1: Append class to file**

```python
class WanStrengthOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanStrengthOptions",
            display_name="Wan Strength Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.CustomInput("WAN_I2V_OPTIONS", "options", optional=True,
                             tooltip="Chain from another options node"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in high-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in high-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for end frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
            ],
            outputs=[
                io.CustomOutput("WAN_I2V_OPTIONS", display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None,
                high_noise_start_strength=1.0, high_noise_mid_strength=0.8,
                low_noise_start_strength=1.0, low_noise_mid_strength=0.2,
                low_noise_end_strength=1.0):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "high_noise_start_strength": high_noise_start_strength,
            "high_noise_mid_strength":   high_noise_mid_strength,
            "low_noise_start_strength":  low_noise_start_strength,
            "low_noise_mid_strength":    low_noise_mid_strength,
            "low_noise_end_strength":    low_noise_end_strength,
        })
        return io.NodeOutput(out)
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('wan_i2v_modular.py').read()); print('OK')"
```

---

## Task 3: Implement `WanContinuationOptions`

**Files:**
- Modify: `wan_i2v_modular.py`

**Step 1: Append class**

```python
class WanContinuationOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanContinuationOptions",
            display_name="Wan Continuation Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.CustomInput("WAN_I2V_OPTIONS", "options", optional=True,
                             tooltip="Chain from another options node"),
                io.Combo.Input("long_video_mode",
                             ["AUTO_CONTINUE", "SVI", "LATENT_CONTINUE"],
                             default="SVI", optional=True,
                             tooltip="AUTO_CONTINUE = pixel-space continuation via motion_frames\nSVI = latent-space continuation via prev_latent\nLATENT_CONTINUE = direct latent injection at frame 0"),
                io.Image.Input("motion_frames", optional=True,
                             tooltip="Last N frames from previous video (AUTO_CONTINUE mode)"),
                io.Latent.Input("prev_latent", optional=True,
                              tooltip="Previous video latent (SVI / LATENT_CONTINUE mode)"),
                io.Int.Input("continue_frames_count", default=5, min=0, max=20, step=1,
                           display_mode=io.NumberDisplay.number, optional=True,
                           tooltip="Number of frames to use from previous video for continuation"),
                io.Float.Input("svi_motion_strength", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="SVI mode motion intensity\n<1.0 = more stable, >1.0 = more dynamic"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1,
                           display_mode=io.NumberDisplay.number, optional=True,
                           tooltip="Frame offset for image sequences spanning multiple clips"),
            ],
            outputs=[
                io.CustomOutput("WAN_I2V_OPTIONS", display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None, long_video_mode="SVI",
                motion_frames=None, prev_latent=None,
                continue_frames_count=5, svi_motion_strength=1.0,
                video_frame_offset=0):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "long_video_mode":       long_video_mode,
            "motion_frames":         motion_frames,
            "prev_latent":           prev_latent,
            "continue_frames_count": continue_frames_count,
            "svi_motion_strength":   svi_motion_strength,
            "video_frame_offset":    video_frame_offset,
        })
        return io.NodeOutput(out)
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('wan_i2v_modular.py').read()); print('OK')"
```

---

## Task 4: Implement `WanClipVisionOptions`

**Files:**
- Modify: `wan_i2v_modular.py`

**Step 1: Append class**

```python
class WanClipVisionOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanClipVisionOptions",
            display_name="Wan Clip Vision Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.CustomInput("WAN_I2V_OPTIONS", "options", optional=True,
                             tooltip="Chain from another options node"),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                        tooltip="CLIP vision embedding for start frame (Wan 2.1 FLF only)"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                        tooltip="CLIP vision embedding for middle frame (Wan 2.1 FLF only)"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                        tooltip="CLIP vision embedding for end frame (Wan 2.1 FLF only)"),
            ],
            outputs=[
                io.CustomOutput("WAN_I2V_OPTIONS", display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None,
                clip_vision_start_image=None,
                clip_vision_middle_image=None,
                clip_vision_end_image=None):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "clip_vision_start_image":  clip_vision_start_image,
            "clip_vision_middle_image": clip_vision_middle_image,
            "clip_vision_end_image":    clip_vision_end_image,
        })
        return io.NodeOutput(out)
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('wan_i2v_modular.py').read()); print('OK')"
```

---

## Task 5: Implement `WanI2VBase`

This is the main node. Its execute() is based on `wan_advanced_i2v.py`'s execute(), but unpacks settings from the options dict instead of taking them as direct inputs. It handles all 4 long_video_mode values.

**Files:**
- Modify: `wan_i2v_modular.py`

**Step 1: Append class**

```python
class WanI2VBase(io.ComfyNode):
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanI2VBase",
            display_name="Wan I2V",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                # Core
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16,
                           display_mode=io.NumberDisplay.number,
                           tooltip="Width of the generated video in pixels"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16,
                           display_mode=io.NumberDisplay.number,
                           tooltip="Height of the generated video in pixels"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4,
                           display_mode=io.NumberDisplay.number,
                           tooltip="Total number of frames in the generated video"),
                io.Int.Input("batch_size", default=1, min=1, max=4096,
                           display_mode=io.NumberDisplay.number,
                           tooltip="Number of videos to generate"),
                # Frame images
                io.Image.Input("start_image", optional=True,
                             tooltip="First frame reference image (anchor)"),
                io.Image.Input("middle_image", optional=True,
                             tooltip="Middle frame reference image for better temporal consistency"),
                io.Image.Input("end_image", optional=True,
                             tooltip="Last frame reference image (target ending)"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0,
                             step=0.01, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Temporal position of middle frame (0.0 = start, 1.0 = end)"),
                # Frame enables
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                               tooltip="Enable start frame conditioning"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                               tooltip="Enable middle frame conditioning"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                               tooltip="Enable end frame conditioning"),
                # Mode
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True,
                             tooltip="NORMAL = all frames condition both stages\nSINGLE_PERSON = only start frame conditions low-noise stage"),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0,
                             step=0.05, round=0.01,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Enhances motion between reference frames using spatial gradients\n1.0 = disabled, >1.0 = stronger repulsion in high-motion areas\nOnly affects high-noise stage"),
                # Options chain
                io.CustomInput("WAN_I2V_OPTIONS", "options", optional=True,
                             tooltip="Connect Wan Strength Options, Wan Continuation Options, or Wan Clip Vision Options"),
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
                start_image=None, middle_image=None, end_image=None,
                middle_frame_ratio=0.5,
                enable_start_frame=True, enable_middle_frame=True, enable_end_frame=True,
                mode="NORMAL", structural_repulsion_boost=1.0,
                options=None):

        # --- Unpack options dict ---
        opts = dict(OPTIONS_DEFAULTS)
        if options:
            opts.update(options)

        high_noise_start_strength = opts["high_noise_start_strength"]
        high_noise_mid_strength   = opts["high_noise_mid_strength"]
        low_noise_start_strength  = opts["low_noise_start_strength"]
        low_noise_mid_strength    = opts["low_noise_mid_strength"]
        low_noise_end_strength    = opts["low_noise_end_strength"]
        long_video_mode           = opts["long_video_mode"]
        motion_frames             = opts["motion_frames"]
        prev_latent               = opts["prev_latent"]
        continue_frames_count     = opts["continue_frames_count"]
        svi_motion_strength       = opts["svi_motion_strength"]
        video_frame_offset        = opts["video_frame_offset"]
        clip_vision_start_image   = opts["clip_vision_start_image"]
        clip_vision_middle_image  = opts["clip_vision_middle_image"]
        clip_vision_end_image     = opts["clip_vision_end_image"]

        # Delegate to the shared implementation (imported from wan_advanced_i2v)
        from .wan_advanced_i2v import WanAdvancedI2V
        return WanAdvancedI2V.execute(
            positive=positive, negative=negative, vae=vae,
            width=width, height=height, length=length, batch_size=batch_size,
            mode=mode,
            start_image=start_image, middle_image=middle_image, end_image=end_image,
            middle_frame_ratio=middle_frame_ratio,
            motion_frames=motion_frames,
            video_frame_offset=video_frame_offset,
            long_video_mode=long_video_mode,
            continue_frames_count=continue_frames_count,
            high_noise_start_strength=high_noise_start_strength,
            high_noise_mid_strength=high_noise_mid_strength,
            low_noise_start_strength=low_noise_start_strength,
            low_noise_mid_strength=low_noise_mid_strength,
            low_noise_end_strength=low_noise_end_strength,
            structural_repulsion_boost=structural_repulsion_boost,
            clip_vision_start_image=clip_vision_start_image if enable_start_frame else None,
            clip_vision_middle_image=clip_vision_middle_image if enable_middle_frame else None,
            clip_vision_end_image=clip_vision_end_image if enable_end_frame else None,
            enable_start_frame=enable_start_frame,
            enable_middle_frame=enable_middle_frame,
            enable_end_frame=enable_end_frame,
            svi_motion_strength=svi_motion_strength,
            prev_latent=prev_latent,
        )
```

**Note:** Delegating to `WanAdvancedI2V.execute()` means zero code duplication. All bug fixes and future improvements to `wan_advanced_i2v.py` automatically apply here.

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('wan_i2v_modular.py').read()); print('OK')"
```

---

## Task 6: Add node registration to `__init__.py`

**Files:**
- Modify: `__init__.py`

**Step 1: Add import block after the `HAS_SVI_PRO_ADVANCED` try/except**

```python
HAS_MODULAR = False
try:
    from .wan_i2v_modular import (
        WanI2VBase,
        WanStrengthOptions,
        WanContinuationOptions,
        WanClipVisionOptions,
    )
    HAS_MODULAR = True
except ImportError:
    print("wan_i2v_modular.py not found")
```

**Step 2: Add nodes to `get_node_list()`**

```python
        if HAS_MODULAR:
            nodes.extend([
                WanI2VBase,
                WanStrengthOptions,
                WanContinuationOptions,
                WanClipVisionOptions,
            ])
```

**Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('__init__.py').read()); print('OK')"
```

---

## Task 7: Verify `io.CustomInput` / `io.CustomOutput` API

**Important:** Check whether `comfy_api.latest.io` exposes `CustomInput` and `CustomOutput`. If not, the correct approach for custom types is:

```python
# Instead of io.CustomInput("WAN_I2V_OPTIONS", "options", optional=True)
# Use:
io.Any.Input("options", optional=True)
```

Check by inspecting the existing nodes or running:

```bash
python -c "from comfy_api.latest import io; print([x for x in dir(io) if 'Custom' in x or 'Any' in x])"
```

Adjust all `io.CustomInput` / `io.CustomOutput` calls in `wan_i2v_modular.py` based on what the API actually provides.

---

## Task 8: Final syntax check and commit

**Step 1: Check all files**

```bash
for f in wan_i2v_modular.py __init__.py; do
  python -c "import ast; ast.parse(open('$f').read()); print('$f: OK')"
done
```

**Step 2: Commit**

```bash
git add wan_i2v_modular.py __init__.py docs/plans/2026-04-03-wan-i2v-modular-nodes.md
git commit -m "feat: add modular Wan I2V nodes with chainable options"
git push
```

---

## Usage Example

```
[Wan Continuation Options] ──options──┐
[Wan Strength Options]     ──options──┤
                                      ▼
                              [Wan I2V Base] ──► positive_high / positive_low / negative / latent
```

For simple I2V with no continuation, just use `[Wan I2V Base]` alone with no options connected.
