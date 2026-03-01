import argparse
import csv
import re
import time
from pathlib import Path

import cv2

SHAPES = ["segitiga", "kotak", "lingkaran"]
COLORS = ["merah", "kuning", "biru"]
SIZES = ["kecil", "sedang", "besar"]


def build_class_map() -> list[tuple[int, str]]:
    class_map: list[tuple[int, str]] = []
    class_id = 1
    for shape in SHAPES:
        for color in COLORS:
            for size in SIZES:
                class_map.append((class_id, f"{shape}_{color}_{size}"))
                class_id += 1
    return class_map


def write_class_map_csv(path: Path, class_map: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class_name"])
        writer.writerows(class_map)


def get_next_index(class_dir: Path, class_name: str) -> int:
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d{{4,}})\.jpg$")
    max_idx = 0
    for image_path in class_dir.glob(f"{class_name}_*.jpg"):
        match = pattern.match(image_path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def save_frame(frame, class_dir: Path, class_name: str, next_idx: int) -> tuple[Path, int]:
    file_path = class_dir / f"{class_name}_{next_idx:04d}.jpg"
    ok = cv2.imwrite(str(file_path), frame)
    if not ok:
        raise RuntimeError(f"Gagal menyimpan gambar: {file_path}")
    return file_path, next_idx + 1


def prompt_choice(label: str, options: list[str], preset: str | None) -> str:
    if preset is not None:
        value = preset.strip().lower()
        if value in options:
            return value
        raise ValueError(f"{label} tidak valid: {preset}")

    print(f"\nPilih {label}:")
    for idx, item in enumerate(options, start=1):
        print(f"{idx}. {item}")

    while True:
        raw = input(f"Masukkan nomor {label} (1-{len(options)}): ").strip()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index]
        print("Input tidak valid, coba lagi.")


def choose_class(class_map: list[tuple[int, str]], shape: str | None, color: str | None, size: str | None) -> tuple[int, str]:
    selected_shape = prompt_choice("bentuk", SHAPES, shape)
    selected_color = prompt_choice("warna", COLORS, color)
    selected_size = prompt_choice("ukuran", SIZES, size)

    class_name = f"{selected_shape}_{selected_color}_{selected_size}"
    name_to_id = {name: cid for cid, name in class_map}
    if class_name not in name_to_id:
        raise RuntimeError(f"Kelas tidak ditemukan: {class_name}")

    return name_to_id[class_name], class_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture dataset otomatis per kombinasi bentuk/warna/ukuran")
    parser.add_argument("--shape", type=str, default=None, help="bentuk: segitiga|kotak|lingkaran")
    parser.add_argument("--color", type=str, default=None, help="warna: merah|kuning|biru")
    parser.add_argument("--size", type=str, default=None, help="ukuran: kecil|sedang|besar")
    parser.add_argument("--camera-index", type=int, default=0, help="Index kamera (default: 0)")
    parser.add_argument("--interval-ms", type=int, default=300, help="Interval burst capture dalam milidetik")
    parser.add_argument("--width", type=int, default=1280, help="Lebar frame")
    parser.add_argument("--height", type=int, default=720, help="Tinggi frame")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"), help="Root folder dataset raw")
    parser.add_argument("--class-map-path", type=Path, default=Path("data/class_map.csv"), help="Lokasi class map csv")
    args = parser.parse_args()

    class_map = build_class_map()
    write_class_map_csv(args.class_map_path, class_map)

    selected_class_id, class_name = choose_class(class_map, args.shape, args.color, args.size)

    class_dir = args.data_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    next_idx = get_next_index(class_dir, class_name)
    total_saved = next_idx - 1

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka. Cek camera-index atau koneksi kamera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("\nKontrol:")
    print("  c : toggle burst capture (ON/OFF)")
    print("  s : simpan 1 frame")
    print("  q : keluar")
    print(f"\nClass ID: {selected_class_id}")
    print(f"Class Name: {class_name}")
    print(f"Output dir: {class_dir}\n")

    burst_mode = False
    last_capture = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame dari kamera.")
                break

            now = time.time()
            if burst_mode and (now - last_capture) * 1000 >= args.interval_ms:
                file_path, next_idx = save_frame(frame, class_dir, class_name, next_idx)
                total_saved += 1
                last_capture = now
                print(f"Saved: {file_path.name}")

            status = "ON" if burst_mode else "OFF"
            lines = [
                f"Class: {selected_class_id} - {class_name}",
                f"Saved: {total_saved}",
                f"Burst: {status} ({args.interval_ms} ms)",
                "Keys: c=burst  s=single  q=quit",
            ]

            y = 30
            for text in lines:
                cv2.putText(
                    frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y += 30

            cv2.imshow("Dataset Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                burst_mode = not burst_mode
                print(f"Burst mode: {'ON' if burst_mode else 'OFF'}")
            if key == ord("s"):
                file_path, next_idx = save_frame(frame, class_dir, class_name, next_idx)
                total_saved += 1
                print(f"Saved: {file_path.name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()