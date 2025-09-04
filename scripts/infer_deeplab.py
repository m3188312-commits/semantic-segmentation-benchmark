from pathlib import Path
import torch, numpy as np
from PIL import Image
import torchvision.transforms as T

from models.deeplab.model   import build_model
from models.deeplab.dataset import CLASS_RGB  # mapping {(r,g,b): idx}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Adjust to your actual weights filename ===
CKPT   = Path('scripts/deeplab_train.pth')

LIST   = Path('data/unlabeled_100_list.txt')            # one path per line (full or filename)
BASE   = Path('data/unlabeled')                        # used if LIST has bare filenames

OUT_MASK_ID  = Path('outputs/unlabeled_preds/deeplab/masks')      # uint8 [0..7]
OUT_MASK_VIS = Path('outputs/unlabeled_preds/deeplab/masks_vis')  # RGB visualization

TARGET_SIZE = (512, 512)  # (width, height)

tfm = T.Compose([
    T.Resize(TARGET_SIZE, interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# inverse palette: class_idx -> (r,g,b)
INV_MAP = {cls: rgb for rgb, cls in CLASS_RGB.items()}

def colorize_ids(idmask: np.ndarray) -> np.ndarray:
    """Map class IDs (H,W) -> RGB (H,W,3) using CLASS_RGB palette."""
    h, w = idmask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in INV_MAP.items():
        rgb[idmask == cls] = col
    return rgb

def read_list_bom_safe(path: Path) -> list[str]:
    # utf-8-sig strips BOM if present
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    return [ln.strip() for ln in lines if ln.strip()]

def main():
    OUT_MASK_ID.mkdir(parents=True, exist_ok=True)
    OUT_MASK_VIS.mkdir(parents=True, exist_ok=True)

    # Load model weights
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    model = build_model(device=DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # Read list; resolve to full paths (use BASE if entries are bare filenames)
    raw = read_list_bom_safe(LIST)
    paths = []
    for entry in raw:
        p = Path(entry)
        paths.append(p if p.is_absolute() or p.parts[0] == 'dataset' else (BASE / p))

    for p in paths:
        if not p.exists():
            print(f"[WARN] missing image: {p} — skipping")
            continue

        img = Image.open(p).convert('RGB')
        x = tfm(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)['out']              # (1, C, H, W)
            pred   = torch.argmax(logits, 1)[0].cpu().numpy().astype(np.uint8)  # (H, W) in [0..7]

        # Save ID mask (for pipeline)
        id_path = OUT_MASK_ID / (p.stem + '.png')
        Image.fromarray(pred, mode='L').save(id_path)

        # Save color visualization (for human check)
        vis = colorize_ids(pred)
        vis_path = OUT_MASK_VIS / (p.stem + '.png')
        Image.fromarray(vis, mode='RGB').save(vis_path)

        print(f"saved id:  {id_path}")
        print(f"saved vis: {vis_path}")

if __name__ == "__main__":
    main()
