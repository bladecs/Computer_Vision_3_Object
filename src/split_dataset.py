import argparse
import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(class_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            images.append(path)
    return sorted(images)


def compute_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    used = train_count + val_count
    test_count = total - used

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count == 0:
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
            test_count = 1

    return train_count, val_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset raw ke train/val/test per kelas")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Folder data raw")
    parser.add_argument("--out-root", type=Path, default=Path("data/split"), help="Folder output split")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Rasio train")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Rasio val")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Rasio test")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--clear", action="store_true", help="Hapus folder split sebelumnya")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("Jumlah rasio train+val+test harus 1.0")

    if not args.raw_root.exists():
        raise FileNotFoundError(f"Folder raw tidak ditemukan: {args.raw_root}")

    class_dirs = [p for p in args.raw_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"Tidak ada folder kelas di: {args.raw_root}")

    if args.clear and args.out_root.exists():
        shutil.rmtree(args.out_root)

    random.seed(args.seed)

    total_all = 0
    print("Memulai split dataset...\n")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = collect_images(class_dir)
        if not images:
            print(f"[SKIP] {class_name}: tidak ada gambar")
            continue

        random.shuffle(images)
        train_count, val_count, test_count = compute_split_counts(
            total=len(images),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:train_count + val_count + test_count]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, split_images in splits.items():
            target_dir = args.out_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                dst = target_dir / src.name
                shutil.copy2(src, dst)

        total_class = len(images)
        total_all += total_class
        print(
            f"[OK] {class_name}: total={total_class}, "
            f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}"
        )

    print(f"\nSelesai. Total gambar diproses: {total_all}")
    print(f"Output split: {args.out_root}")


if __name__ == "__main__":
    main()