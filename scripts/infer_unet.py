from pathlib import Path
import torch, numpy as np
from PIL import Image
import torchvision.transforms as T

from models.unet_no_patches.model   import build_pretrained_unet as build_model
from models.unet_no_patches.dataset import COMMON_SIZE
from models.deeplab.dataset         import CLASS_RGB  # reuse palette

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = Path('scripts/unet_train.pth')    # adjust if needed
LIST   = Path('data/unlabeled_100_list.txt')
BASE   = Path('dataset/unlabeled/image')

OUT_MASK_ID  = Path('outputs/unlabeled_preds/unet/masks')
OUT_MASK_VIS = Path('outputs/unlabeled_preds/unet/masks_vis')

tfm = T.Compose([
    T.Resize(COMMON_SIZE, interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

INV_MAP = {cls: rgb for rgb, cls in CLASS_RGB.items()}

def colorize_ids(idmask: np.ndarray) -> np.ndarray:
    h, w = idmask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in INV_MAP.items():
        rgb[idmask == cls] = col
    return rgb

def read_list_bom_safe(path: Path) -> list[str]:
    return path.read_text(encoding='utf-8-sig').splitlines()

def main():
    OUT_MASK_ID.mkdir(parents=True, exist_ok=True)
    OUT_MASK_VIS.mkdir(parents=True, exist_ok=True)

    model = build_model(device=DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    lines = [ln.strip() for ln in read_list_bom_safe(LIST) if ln.strip()]
    paths = [ (Path(ln) if ln.startswith('dataset') else (BASE / ln)) for ln in lines ]

    for p in paths:
        if not p.exists():
            print(f"[WARN] missing image: {p}"); continue
        img = Image.open(p).convert('RGB')
        x   = tfm(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)                             # (1,C,H,W)
            pred   = torch.argmax(logits, 1)[0].cpu().numpy().astype(np.uint8)

        id_path  = OUT_MASK_ID / (p.stem + '.png')
        vis_path = OUT_MASK_VIS / (p.stem + '.png')
        Image.fromarray(pred, 'L').save(id_path)
        Image.fromarray(colorize_ids(pred), 'RGB').save(vis_path)
        print("saved", id_path, "|", vis_path)

if __name__ == "__main__":
    main()
