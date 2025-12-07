from pathlib import Path
import time
from typing import List, Dict, Any
from itertools import product
import numpy as np
import cv2

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.utils import load_image


def load_pipeline(
    device: str = "cuda",
    torch_dtype=torch.float16,
    controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0",
    vae_id: str = "stabilityai/sdxl-vae",
    base_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype, use_safetensors=True)
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device, dtype=torch_dtype)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def prepare_image(path: str, size: int = 1024) -> Image.Image:
    img = load_image(path).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)

def next_experiment_dir(root: str = "media/experiments") -> (Path, int):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("experiment")]
    max_n = -1
    for p in existing:
        try:
            max_n = max(max_n, int(p.name.replace("experiment", "")))
        except ValueError:
            continue
    new_n = max_n + 1
    exp_dir = root_path / f"experiment{new_n}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, new_n

def load_and_resize(path: str, size: int = 1024) -> (Image.Image, Image.Image):
    img = load_image(path).convert("RGB")
    return img, img.resize((size, size), Image.LANCZOS)

def prepare_control_image(path: str, size: int = 1024, low: int = 100, high: int = 200, canny_dir: Path = Path("media/inputs/canny")) -> (Image.Image, Image.Image, Image.Image):
    control_raw, control_resized = load_and_resize(path, size=size)
    canny_dir.mkdir(parents=True, exist_ok=True)
    control_path = Path(path)
    canny_path = canny_dir / control_path.name
    if canny_path.exists():
        canny_img = Image.open(canny_path).convert("RGB").resize((size, size), Image.LANCZOS)
    else:
        canny = cv2.Canny(np.array(control_resized), low, high)
        canny = np.concatenate([canny[:, :, None]] * 3, axis=2)
        canny_img = Image.fromarray(canny)
        canny_img.save(canny_path)
    return control_raw, control_resized, canny_img

def run_experiments(
    experiments: List[Dict[str, Any]],
    output_dir: str = None,
    device: str = "cuda",
    experiment_seed: int = 42,
):
    if output_dir:
        exp_dir, exp_num = Path(output_dir), None
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir, exp_num = next_experiment_dir()
    inputs_dir = exp_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(device=device)
    generator = torch.Generator(device=device).manual_seed(experiment_seed)
    gen_times: List[float] = []
    t0_total = time.time()
    input_saved, control_saved = set(), set()
    for idx, cfg in enumerate(experiments, start=1):
        prompt = cfg["prompt"]
        negative_prompt = cfg.get("negative_prompt", "low quality, blurry, distorted")
        init_raw, init_resized = load_and_resize(cfg["input_image"])
        ctrl_raw, ctrl_resized, control_image = prepare_control_image(cfg["control_image"])
        num_steps = cfg.get("num_inference_steps", 30)
        guidance_scale = cfg["guidance_scale"]
        strength = cfg["strength"]
        conditioning_scale = cfg["controlnet_conditioning_scale"]

        input_key = Path(cfg["input_image"]).resolve()
        control_key = Path(cfg["control_image"]).resolve()
        raw_in_path = inputs_dir / f"raw_input_{input_key.stem}.png"
        resized_in_path = inputs_dir / f"resized_input_{input_key.stem}.png"
        raw_ctrl_path = inputs_dir / f"raw_control_{control_key.stem}.png"
        resized_ctrl_path = inputs_dir / f"resized_control_{control_key.stem}.png"
        canny_path = inputs_dir / f"canny_control_{control_key.stem}.png"

        if input_key not in input_saved:
            if not raw_in_path.exists():
                init_raw.save(raw_in_path)
            if not resized_in_path.exists():
                init_resized.save(resized_in_path)
            input_saved.add(input_key)
        if control_key not in control_saved:
            if not raw_ctrl_path.exists():
                ctrl_raw.save(raw_ctrl_path)
            if not resized_ctrl_path.exists():
                ctrl_resized.save(resized_ctrl_path)
            if not canny_path.exists():
                control_image.save(canny_path)
            control_saved.add(control_key)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_resized,
            control_image=control_image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        )
        duration = time.time() - t0
        gen_times.append(duration)

        out_image = result.images[0]
        fname = f"gen{idx:02d}_gs{guidance_scale}_str{strength}_cond{conditioning_scale}.png"
        out_path = exp_dir / fname
        out_image.save(out_path)
        avg_time = sum(gen_times) / len(gen_times)
        print(f"[{idx}/{len(experiments)}] saved {out_path} in {duration:.2f}s (avg {avg_time:.2f}s)")

    total = time.time() - t0_total
    avg = sum(gen_times) / len(gen_times) if gen_times else 0.0
    print(f"Experiment completed in {total:.2f}s. Avg/gen: {avg:.2f}s")

    # summarize parameter values (unique lists)
    all_keys = set().union(*experiments)
    param_values = {
        k: sorted({str(e.get(k)) for e in experiments}) for k in all_keys
    }
    input_rel_paths = [str(p.relative_to(exp_dir)) for p in sorted(inputs_dir.iterdir())]

    readme_path = exp_dir / "readme.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("Experiment Summary\n")
        if exp_num is not None:
            f.write(f"Experiment number: {exp_num}\n")
        f.write(f"Seed: {experiment_seed}\n")
        f.write(f"Total generations: {len(experiments)}\n")
        f.write(f"Total time: {total:.2f}s, Avg/gen: {avg:.2f}s\n\n")
        f.write("Parameters used (unique values):\n")
        for k in sorted(param_values.keys()):
            f.write(f"- {k}: {param_values[k]}\n")
        f.write("\nInput files (relative):\n")
        for p in input_rel_paths:
            f.write(f"- {p}\n")

def expand_path_entries(entries: List[str]) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    expanded: List[str] = []
    for e in entries:
        p = Path(e)
        if p.is_dir():
            files = sorted([str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts])
            if not files:
                raise ValueError(f"No image files found in directory: {p}")
            expanded.extend(files)
        else:
            expanded.append(str(p))
    return expanded

def build_experiments_from_lists(param_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    required = [
        "input_image",
        "control_image",
        "prompt",
        "strength",
        "controlnet_conditioning_scale",
        "guidance_scale",
    ]
    optional = ["negative_prompt", "num_inference_steps"]
    missing = [k for k in required if k not in param_lists]
    if missing:
        raise ValueError(f"Missing required parameter lists: {missing}")
    # expand directory entries for image params
    param_lists = dict(param_lists)
    param_lists["input_image"] = expand_path_entries(param_lists["input_image"])
    param_lists["control_image"] = expand_path_entries(param_lists["control_image"])
    for k in required:
        if not param_lists[k]:
            raise ValueError(f"Required list '{k}' must not be empty.")
    keys = required + [k for k in optional if k in param_lists]
    value_lists = [param_lists[k] for k in keys]
    experiments: List[Dict[str, Any]] = []
    for combo in product(*value_lists):
        cfg = dict(zip(keys, combo))
        experiments.append(cfg)
    return experiments

if __name__ == "__main__":
    param_lists = {
        "input_image": ["media/inputs/google-earth"],
        "control_image": ["media/inputs/canny"],
        "prompt": [
            "highly detailed satellite image from Google Earth",
        ],
        "strength": [0.4],
        "controlnet_conditioning_scale": [1.0],
        "guidance_scale": [3.0],
        # Optional lists participate in the cartesian product if provided:
        "negative_prompt": ["low quality, blurry, distorted"],
        "num_inference_steps": [30],
    }
    experiments = build_experiments_from_lists(param_lists)
    run_experiments(experiments, device="cuda")