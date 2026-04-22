"""
organizar_imagenes.py
=====================
Consolida las imágenes de la carpeta "imagenes con polipos destacados"
en dos carpetas bien sincronizadas:

    output/original/   ← imágenes originales (endoscopia)
    output/masks/      ← máscaras binarias correspondientes

Fuentes y convención de nombres resultante
------------------------------------------
  Set 1 → Original/ + Ground Truth/   (612 pares .png numéricos)
           → set1_0001.png … set1_0612.png

  Set 2 → images/ + masks/            (196 pares .png numéricos)
           → set2_0001.png … set2_0196.png

  Set 3 → images_2/ + masks_2/        (pares .jpg con nombres hash)
           Solo se copian los ficheros que tienen pareja en AMBAS carpetas.
           → set3_0001.jpg … set3_NNNN.jpg

Requisitos: Python 3.6+  (solo stdlib, sin dependencias externas)
"""

import shutil
from pathlib import Path


# ─── Rutas base ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "data" / "imagenes_cancer" / "Polipos" / "imagenes con polipos destacados"

SRC_ORIGINAL   = BASE / "Original"
SRC_GROUNDTRUTH = BASE / "Ground Truth"
SRC_IMAGES     = BASE / "images"
SRC_MASKS      = BASE / "masks"
SRC_IMAGES2    = BASE / "images_2"
SRC_MASKS2     = BASE / "masks_2"

OUT_ORIGINAL = BASE / "original"
OUT_MASKS    = BASE / "masks"


def ensure_dirs():
    OUT_ORIGINAL.mkdir(parents=True, exist_ok=True)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def copy_numeric_pairs(src_orig: Path, src_mask: Path,
                       prefix: str, ext: str = ".png") -> int:
    """
    Copia pares de imágenes cuyo nombre es numérico (ej. 1.png, 2.png…).
    Devuelve el número de pares copiados.
    """
    # Obtener los stems (sin extensión) que existen en AMBAS carpetas
    orig_stems = {f.stem for f in src_orig.glob(f"*{ext}")}
    mask_stems = {f.stem for f in src_mask.glob(f"*{ext}")}
    common = orig_stems & mask_stems

    # Ordenar numéricamente para que el renombrado sea predecible
    try:
        sorted_stems = sorted(common, key=lambda s: int(s))
    except ValueError:
        sorted_stems = sorted(common)

    count = 0
    for i, stem in enumerate(sorted_stems, start=1):
        new_name = f"{prefix}{i:04d}{ext}"

        shutil.copy2(src_orig / f"{stem}{ext}",  OUT_ORIGINAL / new_name)
        shutil.copy2(src_mask / f"{stem}{ext}",  OUT_MASKS    / new_name)
        count += 1

    return count


def copy_hash_pairs(src_orig: Path, src_mask: Path, prefix: str) -> int:
    """
    Copia pares de imágenes cuyo nombre es un hash arbitrario.
    Solo procesa los ficheros cuyo nombre (sin extensión) aparece en AMBAS
    carpetas. La extensión puede variar (.jpg, .png…); se compara por stem.
    Devuelve el número de pares copiados.
    """
    # Construir diccionario stem → Path para cada carpeta
    def stem_map(folder: Path):
        return {f.stem: f for f in folder.iterdir() if f.is_file()}

    orig_map = stem_map(src_orig)
    mask_map = stem_map(src_mask)
    common   = sorted(set(orig_map.keys()) & set(mask_map.keys()))

    count = 0
    for i, stem in enumerate(common, start=1):
        orig_file = orig_map[stem]
        mask_file = mask_map[stem]

        new_name_orig = f"{prefix}{i:04d}{orig_file.suffix}"
        new_name_mask = f"{prefix}{i:04d}{mask_file.suffix}"

        shutil.copy2(orig_file, OUT_ORIGINAL / new_name_orig)
        shutil.copy2(mask_file, OUT_MASKS    / new_name_mask)
        count += 1

    return count


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Organizador de imágenes — Predictor Cáncer Rectal")
    print("=" * 60)

    ensure_dirs()

    # Set 1: Original ↔ Ground Truth
    print("\n[1/3] Procesando Set 1 (Original / Ground Truth)…")
    n1 = copy_numeric_pairs(SRC_ORIGINAL, SRC_GROUNDTRUTH,
                            prefix="set1_", ext=".png")
    print(f"      {n1} pares copiados → set1_0001.png … set1_{n1:04d}.png")

    # Set 2: images ↔ masks
    print("\n[2/3] Procesando Set 2 (images / masks)…")
    n2 = copy_numeric_pairs(SRC_IMAGES, SRC_MASKS,
                            prefix="set2_", ext=".png")
    print(f"      {n2} pares copiados → set2_0001.png … set2_{n2:04d}.png")

    # Set 3: images_2 ↔ masks_2
    print("\n[3/3] Procesando Set 3 (images_2 / masks_2)…")
    n3 = copy_hash_pairs(SRC_IMAGES2, SRC_MASKS2, prefix="set3_")
    print(f"      {n3} pares copiados → set3_0001.jpg … set3_{n3:04d}.jpg")

    total = n1 + n2 + n3
    print("\n" + "=" * 60)
    print(f"  ✅ Total de pares procesados: {total}")
    print(f"  📁 Salida en: {OUT_ORIGINAL.parent}")
    print("=" * 60)

    # Verificación rápida
    orig_files = set(f.stem for f in OUT_ORIGINAL.iterdir() if f.is_file())
    mask_files = set(f.stem for f in OUT_MASKS.iterdir()    if f.is_file())

    missing_masks = orig_files - mask_files
    missing_origs = mask_files - orig_files

    if missing_masks or missing_origs:
        print("\nADVERTENCIA: hay desincronización!")
        if missing_masks:
            print(f"   Imágenes sin máscara: {len(missing_masks)} ficheros")
        if missing_origs:
            print(f"   Máscaras sin imagen:  {len(missing_origs)} ficheros")
    else:
        print(f"\n Verificación OK — todas las {len(orig_files)} imágenes")
        print("     tienen su máscara correspondiente.")


if __name__ == "__main__":
    main()
