# scripts/make_train_plus_pseudo.py
# Usage:
#   python -m scripts.make_train_plus_pseudo --n 50 --k 3 --mode keep
#   python -m scripts.make_train_plus_pseudo --n 10 --k 2 --mode remove --seed 123

import argparse, random, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET      = PROJECT_ROOT / "dataset"
PSEUDO_RGB   = PROJECT_ROOT / "data" / "pseudo_masks_rgb"   # colorized masks per K
PSEUDO_IMG   = PROJECT_ROOT / "data" / "pseudo_images"      # masked images per K
UNLAB_IMG    = DATASET / "unlabeled" / "image"              # original unlabeled images

IMG_EXTS  = (".jpg", ".png", ".jpeg")
MASK_EXTS = (".png",)  # your masks are saved as PNG

def copy_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def stems_with_exts(root: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    """Return {stem: path} for files under root matching given extensions (last one wins on dupes)."""
    out = {}
    for ext in exts:
        for p in root.glob(f"*{ext}"):
            out[p.stem] = p
    return out

def resolve_unlabeled_image(stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = UNLAB_IMG / f"{stem}{ext}"
        if p.exists(): return p
    return None

def resolve_masked_image(k: int, stem: str) -> Path | None:
    base = PSEUDO_IMG / f"agreement_{k}"
    for ext in IMG_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists(): return p
    return None

def build_split(n: int, k: int, mode: str, seed: int) -> Path:
    assert mode in {"keep","remove"}
    rng = random.Random(seed)

    # ---- Gather base train pairs by STEM intersection
    base_img_dir = DATASET / "train" / "image"
    base_msk_dir = DATASET / "train" / "mask"

    img_map = stems_with_exts(base_img_dir, IMG_EXTS)
    msk_map = stems_with_exts(base_msk_dir, MASK_EXTS)

    common_stems = sorted(set(img_map).intersection(msk_map))
    missing_imgs = sorted(set(msk_map).difference(img_map))
    missing_msks = sorted(set(img_map).difference(msk_map))

    # ---- Destination layout
    split_name = f"train_plus_pseudo_{n}_K{k}_{mode}"
    dst_root   = DATASET / split_name
    dst_img    = dst_root / "image"
    dst_msk    = dst_root / "mask"

    if dst_root.exists():
        shutil.rmtree(dst_root)

    # 1) Copy ONLY matched base pairs (guarantees base parity)
    for stem in common_stems:
        copy_one(img_map[stem], dst_img / img_map[stem].name)
        copy_one(msk_map[stem], dst_msk / msk_map[stem].name)

    print(f"[INFO] Base train pairs copied: {len(common_stems)}")
    if missing_imgs:
        print(f"[WARN] Skipped {len(missing_imgs)} masks without images (e.g., {missing_imgs[:3]}...)")
    if missing_msks:
        print(f"[WARN] Skipped {len(missing_msks)} images without masks (e.g., {missing_msks[:3]}...)")

    # 2) Sample N pseudo-labeled masks for agreement K
    pseudo_mask_dir = PSEUDO_RGB / f"agreement_{k}"
    if not pseudo_mask_dir.exists():
        raise FileNotFoundError(f"Missing pseudo masks: {pseudo_mask_dir}")

    candidates = sorted(pseudo_mask_dir.glob("*.png"))
    if len(candidates) < n:
        raise ValueError(f"Requested N={n} but only {len(candidates)} pseudo masks in {pseudo_mask_dir}")

    sample = rng.sample(candidates, n)

    # 3) Image resolver for pseudo pairs
    if mode == "keep":
        def img_src_for(stem: str) -> Path:
            p = resolve_unlabeled_image(stem)
            if p is None:
                raise FileNotFoundError(f"Unlabeled image for stem='{stem}' not found in {UNLAB_IMG}")
            return p
    else:
        def img_src_for(stem: str) -> Path:
            p = resolve_masked_image(k, stem)
            if p is None:
                raise FileNotFoundError(f"Masked image for stem='{stem}' not found in {PSEUDO_IMG / f'agreement_{k}'}")
            return p

    # 4) Copy pseudo pairs with UNIQUE names to avoid any overwrite
    added = 0
    for mask_path in sample:
        stem = mask_path.stem
        src_img = img_src_for(stem)

        img_dst_name = f"{stem}__pseudo_K{k}{src_img.suffix}"  # keep original ext
        msk_dst_name = f"{stem}__pseudo_K{k}.png"               # mask always png

        copy_one(src_img,   dst_img / img_dst_name)
        copy_one(mask_path, dst_msk / msk_dst_name)
        added += 1

    # 5) Final parity check (must match)
    ni = len(list(dst_img.glob("*")))
    nm = len(list(dst_msk.glob("*")))
    if ni != nm:
        raise RuntimeError(
            f"Image/Mask count mismatch in {dst_root}: image={ni}, mask={nm}. "
            f"Base matched={len(common_stems)}, pseudo added={added}"
        )

    print(f"✅ Built split: {split_name}")
    print(f"   Base matched pairs: {len(common_stems)}")
    print(f"   Added pseudo:       {added} (N={n}, K={k}, mode={mode})")
    print(f"   Final counts:       images={ni}, masks={nm}")
    print(f"   → {dst_img}")
    print(f"   → {dst_msk}")
    return dst_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, required=True, choices=[10,50,100], help="number of pseudo pairs to add")
    ap.add_argument("--k",    type=int, required=True, choices=[2,3,4],     help="agreement threshold")
    ap.add_argument("--mode", choices=["keep","remove"], required=True,     help="'keep' original image or 'remove' (masked)")
    ap.add_argument("--seed", type=int, default=42,                          help="sampling seed for reproducibility")
    args = ap.parse_args()

    build_split(args.n, args.k, args.mode, args.seed)

if __name__ == "__main__":
    main()
