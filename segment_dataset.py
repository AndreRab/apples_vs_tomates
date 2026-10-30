"""Eksport segmentacji Lab + Otsu z notebooka dla całego datasetu."""
from pathlib import Path
import csv
import numpy as np
import cv2
from PIL import Image, ImageOps

def clean_mask(mask):
    mask = mask.astype(np.uint8)*255
    k = max(3, int(round(min(mask.shape)*.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours: cv2.drawContours(out, [max(contours,key=cv2.contourArea)], -1, 255, -1)
    return out > 0

def segment(rgb, method='lab'):
    # Wspólna skala robocza ogranicza koszt i stabilizuje morfologię.
    h,w = rgb.shape[:2]
    scale = min(1, 384/max(h,w))
    small = cv2.resize(rgb, (max(1,round(w*scale)),max(1,round(h*scale))))
    if method == 'hsv':
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        raw = (hsv[:,:,1] > 55) & (hsv[:,:,2] > 35)
    else:
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        b = max(1, round(min(lab.shape[:2])*.04))
        border = np.concatenate([lab[:b].reshape(-1,3), lab[-b:].reshape(-1,3),
                                 lab[:,:b].reshape(-1,3), lab[:,-b:].reshape(-1,3)])
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab-background, axis=2)
        distance = np.uint8(np.clip(distance,0,255))
        _, raw = cv2.threshold(distance,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        raw = raw > 0
    cleaned = clean_mask(raw)
    return cv2.resize(cleaned.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    root = Path(__file__).resolve().parent
    data = root / 'apples_tomatoes'
    output = root / 'segmented_dataset'
    rows = []
    for path in sorted(data.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        relative = path.relative_to(data)
        with Image.open(path) as im:
            rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
        mask = segment(rgb, 'lab')
        # Zachowanie rozszerzenia źródła w nazwie zapobiega kolizjom nazw.
        name = relative.parent / (relative.name + '.png')
        mask_path = output / 'masks' / name
        image_path = output / 'images' / name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        rgba = np.dstack((rgb, mask.astype(np.uint8) * 255))
        Image.fromarray(rgba).save(image_path)
        edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        fraction = float(mask.mean())
        border = float(edge.mean())
        # Kontrola każdego zapisanego wyniku, także kanału przezroczystości.
        with Image.open(mask_path) as saved_mask, Image.open(image_path) as saved_image:
            assert saved_mask.size == saved_image.size == (rgb.shape[1], rgb.shape[0])
            assert set(np.unique(np.array(saved_mask))) <= {0, 255}
            assert np.array_equal(np.array(saved_image)[:, :, 3], mask.astype(np.uint8)*255)
        rows.append(dict(source=str(relative), split=relative.parts[0], label=relative.parts[1],
                         mask=str(mask_path.relative_to(output)), image=str(image_path.relative_to(output)),
                         foreground_fraction=fraction, border_fraction=border,
                         needs_review=fraction < .05 or fraction > .90 or border > .25))
    with (output / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(r['needs_review'] for r in rows)
    (output / 'README.md').write_text(f"""# Segmentacja całego datasetu

Metoda: Lab + Otsu z notebooka task3.ipynb, morfologia i największy komponent.
Parametry identyczne dla train/test, bez dostrajania na teście.

- `images/`: obrazy PNG RGBA z przezroczystym tłem, w oryginalnym rozmiarze.
- `masks/`: maski PNG, 255 = obiekt, 0 = tło.
- W obu folderach zachowano podział train/test i apples/tomatoes.
- Nazwa `oryginal.jpeg.png` zachowuje pełną nazwę pliku źródłowego.
- `manifest.csv`: mapowanie źródeł na wyniki oraz diagnostyka.

Przetworzono {len(rows)} obrazów. Heurystyka oznaczyła {flagged} masek do przeglądu.
To segmentacja wstępna: może wybierać tło, pomijać owoce lub zachowywać tylko jeden
z wielu rozłącznych obiektów. Ostrzeżenia nie wykrywają wszystkich błędów.
Duplikaty i oryginalny podział zbioru zostały zachowane.

Ponowne generowanie: `uv run python segment_dataset.py` z folderu third_task.
""")
    print(f'Zapisano i zweryfikowano {len(rows)} masek i {len(rows)} obrazów; do przeglądu: {flagged}')
    for split in ('train', 'test'):
        for label in ('apples', 'tomatoes'):
            print(split, label, sum(r['split'] == split and r['label'] == label for r in rows))

if __name__ == '__main__':
    main()
