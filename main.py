"""
Frame-by-frame video generation using SDXL img2img + ControlNet.

Edit the configuration block below to change inputs, prompts, schedules,
and output paths. No CLI parsing is used.

Luke Dzwonczyk, Dec 2025
"""

import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

from experiments import load_pipeline


# ---------- User configuration ----------

# One image path, a list of paths, or a directory. If it resolves to multiple
# images, the run cycles through them for however many frames you request.
# If it resolves to one image, the run reuses it for ``NUM_FRAMES`` frames.
INPUTS = "media/inputs/google-earth"

# Single control/canny image path.
CONTROL_IMAGE = "media/inputs/canny/gingko.jpg"

# Total frames to generate (applies to both single-image and folder modes).
NUM_FRAMES = 8
# google_earth has 88 images

# Prompt settings.
PROMPT = "highly detailed satellite image with the outline of a gingko leaf in the center"
NEGATIVE_PROMPT = "low quality, blurry, distorted"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0

# Parameter schedules for strength and ControlNet conditioning scale.
# Supported forms: a single number (fixed), a list (per-frame),
# {"mode": "linear", "start": a, "end": b}, {"mode": "random", "min": a, "max": b}.
# STRENGTH_SCHEDULE = {"mode": "linear", "start": 0.2, "end": 1.0} # img2img denoising strength
STRENGTH_SCHEDULE = 0.4 # fixed img2img denoising strength

CONDITIONING_SCHEDULE = 1.0 # Controlnet Conditioning Strength

# Output resolution (must be multiples of 8).
# IMAGE_SIZE = (1920, 1080)
IMAGE_SIZE = (1360, 768)
# IMAGE_SIZE = (1280, 720)
# IMAGE_SIZE = (1024, 1024) # SDXL was trained on 1024x1024 images, its best to keep it around 1m pixels

# Output locations.
OUTPUT_ROOT = Path("media/videos")
RUN_NAME = None  # If None, auto-increments runX

# Video settings.
FPS = 2
VIDEO_FILENAME = "video.mp4"

# Optional upscaling.
UPSCALE = True
UPSCALE_SIZE = (1920, 1080)

# Canny edge detector thresholds.
CANNY_LOW = 100
CANNY_HIGH = 200


# Reproducibility.
SEED = 42
SCHEDULE_SEED = 123

# Device / dtype.
DEVICE = "cuda"
TORCH_DTYPE = torch.float16


# ---------- Helpers ----------


def ensure_size_multiple_of_8(size):
	w, h = size
	if w % 8 != 0 or h % 8 != 0:
		raise ValueError(f"IMAGE_SIZE must be multiples of 8, got {size}")


def next_run_dir(root=Path("media/videos"), name=None):
	root.mkdir(parents=True, exist_ok=True)
	if name:
		run_dir = root / name
		run_dir.mkdir(parents=True, exist_ok=True)
		return run_dir, None
	existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run")]
	max_n = -1
	for p in existing:
		try:
			max_n = max(max_n, int(p.name.replace("run", "")))
		except ValueError:
			continue
	new_n = max_n + 1
	run_dir = root / f"run{new_n}"
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir, new_n


def load_image_rgb(path):
	return Image.open(path).convert("RGB")


def resize_image(img, size):
	return img.resize(size, Image.LANCZOS)


def resize_cover_crop(img, target_size):
	tw, th = target_size
	w, h = img.size
	scale = max(tw / w, th / h)
	resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
	rw, rh = resized.size
	left = max((rw - tw) // 2, 0)
	top = max((rh - th) // 2, 0)
	return resized.crop((left, top, left + tw, top + th))


def resize_contain_pad(img, target_size, fill=(0, 0, 0)):
	tw, th = target_size
	w, h = img.size
	scale = min(1.0, min(tw / w, th / h))
	new_w, new_h = int(w * scale), int(h * scale)
	resized = img.resize((new_w, new_h), Image.LANCZOS)
	canvas = Image.new("RGB", (tw, th), fill)
	left = (tw - new_w) // 2
	top = (th - new_h) // 2
	canvas.paste(resized, (left, top))
	return canvas


def prepare_init_image(path, size):
	raw = load_image_rgb(path)
	resized = resize_cover_crop(raw, size)
	return raw, resized


def prepare_control_image_cached(
	path,
	size,
	low,
	high,
	cache_dir,

):
	cache_dir.mkdir(parents=True, exist_ok=True)
	raw = load_image_rgb(path)
	resized = resize_contain_pad(raw, size)
	cache_path = cache_dir / path.name
	if cache_path.exists():
		canny_img = load_image_rgb(cache_path)
		if canny_img.size != size:
			canny_img = resize_contain_pad(canny_img, size)
	else:
		canny = cv2_canny(resized, low, high)
		canny_img = Image.fromarray(canny)
		canny_img.save(cache_path)
	return raw, resized, canny_img, cache_path


def cv2_canny(img, low, high):
	arr = np.array(img)
	edges = cv2.Canny(arr, low, high)
	edges = np.concatenate([edges[:, :, None]] * 3, axis=2)
	return edges


def expand_input_entry(entry):
	exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
	p = Path(entry)
	if p.is_dir():
		files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts])
		if not files:
			raise ValueError(f"No image files found in directory: {p}")
		return files
	return [p]


def resolve_inputs(inputs):
	entries = inputs if isinstance(inputs, list) else [inputs]
	resolved = []
	for e in entries:
		resolved.extend(expand_input_entry(e))
	return resolved


def build_schedule(spec, length, rng=None, name="param"):
	if length <= 0:
		raise ValueError("length must be positive")
	if isinstance(spec, (int, float)):
		return [float(spec)] * length
	if isinstance(spec, dict):
		mode = spec.get("mode") or spec.get("type")
		if mode == "linear":
			start = float(spec["start"])
			end = float(spec["end"])
			if length == 1:
				return [start]
			step = (end - start) / (length - 1)
			return [start + step * i for i in range(length)]
		if mode == "random":
			low = float(spec["min"])
			high = float(spec["max"])
			rng = rng or random.Random()
			return [rng.uniform(low, high) for _ in range(length)]
		raise ValueError(f"Unsupported schedule mode for {name}: {spec}")
	if isinstance(spec, (list, tuple)) and not isinstance(spec, (str, bytes, dict)):
		values = list(spec)
		if len(values) == 1:
			return [float(values[0])] * length
		if len(values) != length:
			raise ValueError(f"Schedule for {name} must have length {length}, got {len(values)}")
		return [float(v) for v in values]
	raise ValueError(f"Unsupported schedule format for {name}: {spec}")


def write_summary(path, summary):
	lines = ["Run Summary"]
	for k, v in summary.items():
		lines.append(f"{k}: {v}")
	path.write_text("\n".join(lines), encoding="utf-8")


# ---------- Main generation ----------


def main():
	ensure_size_multiple_of_8(IMAGE_SIZE)

	resolved_inputs = resolve_inputs(INPUTS)
	if not resolved_inputs:
		raise ValueError("No input images found.")

	if len(resolved_inputs) == 1:
		mode = "single_pair"
		frame_count = NUM_FRAMES
		init_sequence = resolved_inputs * frame_count
	else:
		mode = "folder_onecanny"
		frame_count = NUM_FRAMES
		count = len(resolved_inputs)
		init_sequence = [resolved_inputs[i % count] for i in range(frame_count)]

	if frame_count <= 0:
		raise ValueError("frame_count must be positive")

	rng = random.Random(SCHEDULE_SEED) if SCHEDULE_SEED is not None else random.Random()
	strength_values = build_schedule(STRENGTH_SCHEDULE, frame_count, rng, "strength")
	conditioning_values = build_schedule(CONDITIONING_SCHEDULE, frame_count, rng, "conditioning")

	run_dir, run_num = next_run_dir(OUTPUT_ROOT, RUN_NAME)
	frames_dir = run_dir / "frames"
	pre_upscale_dir = run_dir / "pre-upscale" if UPSCALE else None
	inputs_dir = run_dir / "inputs"
	frames_dir.mkdir(parents=True, exist_ok=True)
	if pre_upscale_dir:
		pre_upscale_dir.mkdir(parents=True, exist_ok=True)
	inputs_dir.mkdir(parents=True, exist_ok=True)

	control_path = Path(CONTROL_IMAGE)
	ctrl_raw, ctrl_resized, control_image, cached_canny_path = prepare_control_image_cached(
		control_path, IMAGE_SIZE, CANNY_LOW, CANNY_HIGH, inputs_dir
	)
	ctrl_raw.save(inputs_dir / f"raw_control_{control_path.stem}.png")
	ctrl_resized.save(inputs_dir / f"resized_control_{control_path.stem}.png")

	pipe = load_pipeline(device=DEVICE, torch_dtype=TORCH_DTYPE)
	generator = torch.Generator(device=DEVICE).manual_seed(SEED)

	saved_inputs = set()
	gen_times = []
	t0_total = time.time()

	for idx, init_path in enumerate(init_sequence):
		init_raw, init_resized = prepare_init_image(init_path, IMAGE_SIZE)
		if init_path not in saved_inputs:
			init_raw.save(inputs_dir / f"raw_input_{init_path.stem}.png")
			init_resized.save(inputs_dir / f"resized_input_{init_path.stem}.png")
			saved_inputs.add(init_path)

		strength = strength_values[idx]
		conditioning_scale = conditioning_values[idx]

		t0 = time.time()
		result = pipe(
			prompt=PROMPT,
			negative_prompt=NEGATIVE_PROMPT,
			image=init_resized,
			control_image=control_image,
			num_inference_steps=NUM_INFERENCE_STEPS,
			guidance_scale=GUIDANCE_SCALE,
			strength=strength,
			controlnet_conditioning_scale=conditioning_scale,
			generator=generator,
		)
		duration = time.time() - t0
		gen_times.append(duration)

		out_image = result.images[0]
		frame_path = frames_dir / f"frame_{idx:05d}.png"

		if UPSCALE and pre_upscale_dir:
			pre_path = pre_upscale_dir / f"frame_{idx:05d}.png"
			out_image.save(pre_path)
			saved_path = pre_path
		else:
			out_image.save(frame_path)
			saved_path = frame_path

		avg_time = sum(gen_times) / len(gen_times)
		remaining_time = avg_time * (frame_count - idx - 1)
		print(
			f"[{idx + 1}/{frame_count}] saved {saved_path} in {duration:.2f}s "
			f"(avg {avg_time:.2f}s, remaining {format_duration(remaining_time)}) | "
			f"strength={strength:.3f}, cond={conditioning_scale:.3f}"
		)

	if UPSCALE and pre_upscale_dir:
		del pipe
		torch.cuda.empty_cache()
		upscaler = StableDiffusionXLImg2ImgPipeline.from_pretrained(
			"stabilityai/stable-diffusion-xl-base-1.0",
			torch_dtype=torch.float16,
			use_safetensors=True,
			vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16),
		).to(DEVICE)
		upscaler.enable_xformers_memory_efficient_attention()
		for pre_path in sorted(pre_upscale_dir.glob("frame_*.png")):
			base_image = Image.open(pre_path)
			upscaled = upscaler(
				prompt=PROMPT,
				image=base_image.resize(UPSCALE_SIZE, Image.Resampling.LANCZOS),
				strength=0.28,
				guidance_scale=2.5,
				num_inference_steps=12,
			).images[0]
			upscaled.save(frames_dir / pre_path.name)

	video_path = run_dir / VIDEO_FILENAME
	stitch_video(frames_dir, video_path, FPS)

	total = time.time() - t0_total
	avg = sum(gen_times) / len(gen_times) if gen_times else 0.0

	summary = {
		"mode": mode,
		"run_dir": str(run_dir),
		"frames": frame_count,
		"input_files": [str(p) for p in resolved_inputs],
		"control_image": str(control_path),
		"prompt": PROMPT,
		"negative_prompt": NEGATIVE_PROMPT,
		"num_inference_steps": NUM_INFERENCE_STEPS,
		"guidance_scale": GUIDANCE_SCALE,
		"strength_schedule": strength_values,
		"conditioning_schedule": conditioning_values,
		"image_size": IMAGE_SIZE,
		"fps": FPS,
		"video_file": str(video_path),
		"seed": SEED,
		"schedule_seed": SCHEDULE_SEED,
		"device": DEVICE,
		"dtype": str(TORCH_DTYPE),
		"upscale": UPSCALE,
		"upscale_size": UPSCALE_SIZE if UPSCALE else None,
		"total_time_sec": total,
		"avg_time_sec": avg,
	}
	write_summary(run_dir / "readme.txt", summary)
	print(f"Run complete in {format_duration(total)}. Video: {video_path}")


def stitch_video(frames_dir, video_path, fps):
	cmd = [
		"ffmpeg",
		"-y",
		"-framerate",
		str(fps),
		"-i",
		str(frames_dir / "frame_%05d.png"),
		"-c:v",
		"libx264",
		"-pix_fmt",
		"yuv420p",
		str(video_path),
	]
	print("Stitching video with ffmpeg...")
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if proc.returncode != 0:
		print("ffmpeg failed:", proc.stderr)
	else:
		print(f"Video saved to {video_path}")


def format_duration(seconds):
	seconds = int(max(seconds, 0))
	h, rem = divmod(seconds, 3600)
	m, s = divmod(rem, 60)
	parts = []
	if h:
		parts.append(f"{h}h")
	if h or m:
		parts.append(f"{m}m")
	parts.append(f"{s}s")
	return "".join(parts)


if __name__ == "__main__":
	main()
