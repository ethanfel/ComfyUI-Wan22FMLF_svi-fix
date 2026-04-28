# ComfyUI-WanExtended

> Extended multi-frame reference conditioning and long-video continuation nodes for Wan 2.1/2.2 I2V models.

Built on top of [wallen0322/ComfyUI-Wan22FMLF](https://github.com/wallen0322/ComfyUI-Wan22FMLF), this pack adds a unified advanced node, a modular options system, an improved SVI Pro continuation node, and a structural repulsion boost for enhancing motion between reference frames.

---

## Demo — 4-in-1 Long Video (24fps 720p, 403 frames)

https://github.com/user-attachments/assets/dc1cf2a4-3c6a-4210-a247-e53c2423f776

https://github.com/user-attachments/assets/2f4a5b17-610f-4c5c-9e3f-e5fd5083e762

Two 161-frame segments + one 81-frame segment. Each segment's variation is controlled by a middle frame and prompt. Lip sync applied uniformly at the end.

> This workflow combines: SVI Pro Advanced + First-Middle-Last + VBVR physics LoRA + InfiniTalk lip sync (Painter AV2V). The workflow file is in `example_workflows/`.

---

## Nodes

### Wan Advanced I2V *(recommended starting point)*

A unified node with start/middle/end frame references and all four long-video continuation modes in one place.

**Continuation modes** (`long_video_mode`):

| Mode | Description |
|------|-------------|
| `DISABLED` | Standalone generation, no continuation |
| `AUTO_CONTINUE` | Pixel-space continuation via `motion_frames` |
| `SVI` | Latent-space continuation via `prev_latent` |
| `LATENT_CONTINUE` | Direct latent injection at frame 0 |

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_image` / `middle_image` / `end_image` | — | Reference frames (all optional) |
| `middle_frame_ratio` | 0.5 | Temporal position of middle frame (0.0–1.0) |
| `high_noise_start/mid_strength` | 1.0 / 0.8 | Reference strength in high-noise denoising stage |
| `low_noise_start/mid/end_strength` | 1.0 / 0.2 / 1.0 | Reference strength in low-noise denoising stage |
| `structural_repulsion_boost` | 1.0 | Motion enhancement between reference frames (1.0 = off, up to 2.0) |
| `continue_frames_count` | 5 | Frames pulled from previous video for continuation |
| `svi_motion_strength` | 1.0 | SVI motion intensity (< 1.0 = stable, > 1.0 = dynamic) |
| `mode` | NORMAL | `SINGLE_PERSON`: only start frame conditions low-noise stage — reduces identity drift |

Outputs `trim_latent`, `trim_image`, and `next_offset` for chaining into the next segment.

---

### Wan I2V Base + Option Nodes *(modular system)*

A composable alternative to Wan Advanced I2V. Build your setup by chaining option nodes into the base node.

```
[WanStrengthOptions] ──┐
[WanContinuationOptions] ──┤──► [WanI2VBase]
[WanClipVisionOptions] ──┘
```

**WanStrengthOptions** — configure all noise-stage strengths and middle frame ratio.

**WanContinuationOptions** — configure continuation mode, motion frames, SVI latent, frame count, motion strength.

**WanClipVisionOptions** — attach CLIP vision embeddings per frame.

**WanI2VBase** — base node that takes images, options, and produces conditioning + latent. Same output as Wan Advanced I2V.

Use this when you want to reuse option configurations across multiple base nodes, or when the all-in-one node feels cluttered.

---

### Wan SVI Pro Advanced

Dedicated SVI (Stable Video Infinity) continuation node with extended boost controls for motion dynamics.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `prev_latent` | — | — | Previous video segment latent |
| `overlap_frames` | 4 | 0–128 | Pixel frames pulled from prev_latent |
| `motion_influence` | 1.0 | 0.0–2.0 | Overall weight of the motion transfer. Low-res: increase. High-res: decrease. |
| `motion_boost` | 1.0 | 0.5–3.0 | Amplifies or dampens frame-to-frame differences in the motion latent |
| `detail_boost` | 1.0 | 0.5–4.0 | Adjusts mask decay rate — higher = faster decay = more model freedom |
| `structural_repulsion_boost` | 1.0 | 1.0–2.0 | Motion enhancement between reference frames, high-noise stage only |

**Recommended starting values:**
```
motion_influence: 1.0–1.3 | overlap_frames: 4 | motion_boost: 1.0–1.5 | detail_boost: 1.0–1.5
```

Also accepts `start_image`, `middle_image`, `end_image` and their conditioning strength controls, same as Wan Advanced I2V.

---

### Wan First-Middle-Last Frame

Simple 3-frame reference node (start, middle, end). Good baseline when you don't need continuation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_image` / `middle_image` / `end_image` | — | Reference frames (all optional) |
| `middle_frame_ratio` | 0.5 | Middle frame temporal position |
| `high_noise_mid_strength` | 0.8 | Middle frame strength, high-noise stage |
| `low_noise_start/mid/end_strength` | 1.0 / 0.2 / 1.0 | Per-frame strength, low-noise stage |
| `structural_repulsion_boost` | 1.0 | Motion enhancement (1.0 = off) |
| `mode` | NORMAL | `SINGLE_PERSON` for better identity consistency |

---

### Wan 4-Frame Reference

Four individually positioned reference frames, each with independent per-stage strength controls. Frames 2 and 3 are intermediate and can be disabled.

---

### Wan Multi-Frame Reference

Accepts a batch of images and distributes them across the timeline. Useful for 5+ reference frames.

**`ref_positions` format:**
- Empty: auto-distributes frames evenly
- Ratios `0.0–1.99`: `"0, 0.33, 0.67, 1.0"` — as fraction of video length
- Absolute indices `>= 2`: `"0, 27, 54, 80"` — direct frame positions
- JSON array: `"[0, 0.33, 54, 1.0]"` — mix of both

All positions are snapped to latent-aligned multiples of 4.

---

### Utility Nodes

- **Wan Multi-Image Loader** — load multiple images with server-side storage and browser preview UI
- **Wan Advanced Extract Last Frames** — extract the last N latent frames from a video latent
- **Wan Advanced Extract Last Images** — extract the last N pixel frames from a decoded video

---

## Tips

### Resolution

| Category | Recommended |
|----------|-------------|
| Standard | 480×832, 832×480, 576×1024 |
| High-res | 704×1280, 1280×704 |

Avoid 720×1280 — causes middle-frame flickering.

### Noise stage steps

- 2 high-noise steps is usually enough. More steps increase middle-frame flicker probability.

### Middle frame strength

- Typical: high=0.6–0.8, low=0.2
- Complex scenes (many changes): high=0.6–0.8, low=0.0

### High-variation scenes (transformations, rapid changes)

Use `NORMAL` mode and reduce any LightX2V LoRA weight to ~0.6. `SINGLE_PERSON` mode can suppress strong transitions.

### Start/end frame strength in SVI mode

Keep below 0.5 — higher values can cause gray-out artifacts at splice points.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-WanExtended
```

Restart ComfyUI. No additional dependencies required beyond a standard ComfyUI installation.

---

## License

See [LICENSE](LICENSE).
