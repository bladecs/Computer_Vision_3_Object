import argparse
import json
import socket
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from model_cnn import CustomVisionCNN


def preprocess_bgr(frame_bgr: np.ndarray, image_size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x)


def find_color_contours(frame_bgr: np.ndarray, min_area: int) -> tuple[list[np.ndarray], np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 40], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([160, 70, 40], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    lower_yellow = np.array([18, 70, 40], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    lower_blue = np.array([90, 70, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    red_mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = red_mask | yellow_mask | blue_mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid, mask


def draw_rotated_outline(frame_bgr: np.ndarray, contour: np.ndarray) -> tuple[int, int, int, int]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.polylines(frame_bgr, [box], isClosed=True, color=(0, 255, 255), thickness=2)

    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def crop_with_padding(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> np.ndarray:
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, w_img)
    y2 = min(y + h + pad, h_img)
    return frame_bgr[y1:y2, x1:x2]


def estimate_shape(contour: np.ndarray) -> str:
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return "unknown"

    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    vertices = len(approx)

    area = cv2.contourArea(contour)
    circularity = 0.0
    if peri > 0:
        circularity = float((4 * np.pi * area) / (peri * peri))

    if vertices == 3:
        return "segitiga"
    if vertices == 4:
        return "kotak"
    if circularity >= 0.78:
        return "lingkaran"

    if vertices <= 5:
        return "kotak"
    return "lingkaran"


def estimate_size(area: float, small_max: float, medium_max: float) -> str:
    if area < small_max:
        return "kecil"
    if area < medium_max:
        return "sedang"
    return "besar"


def estimate_size_for_shape(
    shape: str,
    area: float,
    small_max: float,
    medium_max: float,
    triangle_small_max: float | None,
    triangle_medium_max: float | None,
) -> str:
    if shape == "segitiga":
        small_max = triangle_small_max if triangle_small_max is not None else small_max
        medium_max = triangle_medium_max if triangle_medium_max is not None else medium_max
    return estimate_size(area, small_max, medium_max)


def split_model_label(label: str) -> tuple[str, str, str]:
    parts = label.split("_")
    if len(parts) != 3:
        return "unknown", "unknown", "unknown"
    return parts[0], parts[1], parts[2]


def estimate_color(frame_bgr: np.ndarray, contour: np.ndarray) -> str:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)

    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return "unknown"

    hue = float(np.median(pixels[:, 0]))
    sat = float(np.median(pixels[:, 1]))
    val = float(np.median(pixels[:, 2]))
    if sat < 40 or val < 40:
        return "unknown"

    if hue <= 10 or hue >= 160:
        return "merah"
    if 18 <= hue <= 40:
        return "kuning"
    if 90 <= hue <= 140:
        return "biru"
    return "unknown"


def contour_center(contour: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def majority_label(history: deque[str], min_votes: int) -> str | None:
    if not history:
        return None
    label, count = Counter(history).most_common(1)[0]
    if count >= min_votes:
        return label
    return None


def try_connect_tcp(host: str, port: int, timeout: float = 2.0) -> socket.socket | None:
    if not host or port <= 0:
        return None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock
    except OSError:
        return None


def send_tcp_message(sock: socket.socket, payload: dict) -> bool:
    try:
        data = json.dumps(payload, ensure_ascii=True) + "\n"
        sock.sendall(data.encode("utf-8"))
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime test model dengan outline semua objek")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/custom_cnn_all.pth"), help="Path model checkpoint")
    parser.add_argument("--camera-index", type=int, default=0, help="Index kamera")
    parser.add_argument("--min-area", type=int, default=1800, help="Luas kontur minimum")
    parser.add_argument("--pad", type=int, default=10, help="Padding crop")
    parser.add_argument("--conf-threshold", type=float, default=0.60, help="Threshold confidence model")
    parser.add_argument("--small-max-area", type=float, default=7000, help="Batas maksimum area untuk ukuran kecil")
    parser.add_argument("--medium-max-area", type=float, default=14000, help="Batas maksimum area untuk ukuran sedang")
    parser.add_argument("--triangle-small-max-area", type=float, default=None, help="Batas maksimum area segitiga untuk ukuran kecil")
    parser.add_argument("--triangle-medium-max-area", type=float, default=None, help="Batas maksimum area segitiga untuk ukuran sedang")
    parser.add_argument("--vote-window", type=int, default=8, help="Jumlah frame untuk voting label")
    parser.add_argument("--min-votes", type=int, default=4, help="Minimal suara untuk update label stabil")
    parser.add_argument("--max-track-distance", type=float, default=90.0, help="Jarak maksimum asosiasi objek antar frame")
    parser.add_argument("--max-missing", type=int, default=12, help="Batas frame hilang sebelum track dihapus")
    parser.add_argument("--tcp-host", type=str, default="", help="Host penerima TCP (kosong untuk disable)")
    parser.add_argument("--tcp-port", type=int, default=0, help="Port penerima TCP")
    parser.add_argument("--send-delay-seconds", type=float, default=3.0, help="Delay label stabil sebelum dikirim (detik)")
    parser.add_argument("--send-retry-seconds", type=float, default=5.0, help="Interval coba konek ulang TCP (detik)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    idx_to_class_raw = ckpt["idx_to_class"]
    idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)

    image_size = int(ckpt.get("image_size", 128))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])

    model = CustomVisionCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka")

    tracks: dict[int, dict] = {}
    next_track_id = 1
    frame_id = 0
    send_index = 1
    tcp_sock: socket.socket | None = None
    last_tcp_attempt = 0.0

    print("Tekan q untuk keluar")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            contours, _ = find_color_contours(frame, min_area=args.min_area)
            detections = []

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = draw_rotated_outline(frame, contour)
                obj_crop = crop_with_padding(frame, x, y, w, h, pad=args.pad)
                if obj_crop.size == 0:
                    continue

                input_tensor = preprocess_bgr(obj_crop, image_size=image_size, mean=mean, std=std).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)

                conf_value = float(conf.item())
                pred_index = int(pred_idx.item())
                model_label = idx_to_class[pred_index]

                model_shape, model_color, model_size = split_model_label(model_label)
                geo_shape = estimate_shape(contour)
                geo_color = estimate_color(frame, contour)

                if conf_value >= args.conf_threshold and model_shape != "unknown":
                    shape = geo_shape if geo_shape != "unknown" else model_shape
                else:
                    shape = geo_shape if geo_shape != "unknown" else model_shape

                geo_size = estimate_size_for_shape(
                    shape,
                    area,
                    args.small_max_area,
                    args.medium_max_area,
                    args.triangle_small_max_area,
                    args.triangle_medium_max_area,
                )
                color = geo_color if geo_color != "unknown" else model_color
                size = geo_size if geo_size != "unknown" else model_size
                current_label = f"{shape}_{color}_{size}"

                detections.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "area": area,
                        "center": contour_center(contour),
                        "label": current_label,
                        "conf": conf_value,
                    }
                )

            unmatched_tracks = set(tracks.keys())
            for det in detections:
                best_track_id = None
                best_dist = args.max_track_distance
                cx, cy = det["center"]

                for track_id in list(unmatched_tracks):
                    tx, ty = tracks[track_id]["center"]
                    dist = float(np.hypot(cx - tx, cy - ty))
                    if dist < best_dist:
                        best_dist = dist
                        best_track_id = track_id

                if best_track_id is None:
                    best_track_id = next_track_id
                    next_track_id += 1
                    now_mono = time.monotonic()
                    tracks[best_track_id] = {
                        "center": det["center"],
                        "history": deque(maxlen=max(1, args.vote_window)),
                        "stable_label": det["label"],
                        "conf": det["conf"],
                        "last_seen": frame_id,
                        "stable_since": now_mono,
                        "last_sent_label": None,
                        "last_sent_time": 0.0,
                    }
                else:
                    unmatched_tracks.discard(best_track_id)

                track = tracks[best_track_id]
                track["center"] = det["center"]
                track["last_seen"] = frame_id
                track["history"].append(det["label"])
                stable = majority_label(track["history"], args.min_votes)
                if stable is not None:
                    if stable != track["stable_label"]:
                        track["stable_label"] = stable
                        track["stable_since"] = time.monotonic()
                track["conf"] = det["conf"]

                det["track_id"] = best_track_id
                det["stable_label"] = track["stable_label"]
                det["stable_conf"] = track["conf"]

            stale_ids = [tid for tid, t in tracks.items() if frame_id - t["last_seen"] > args.max_missing]
            for tid in stale_ids:
                del tracks[tid]

            if args.tcp_host and args.tcp_port > 0:
                now_mono = time.monotonic()
                if tcp_sock is None and (now_mono - last_tcp_attempt) >= args.send_retry_seconds:
                    last_tcp_attempt = now_mono
                    tcp_sock = try_connect_tcp(args.tcp_host, args.tcp_port)
                    if tcp_sock is None:
                        print("Gagal konek TCP. Akan coba lagi.")

                if tcp_sock is not None:
                    for track_id, track in tracks.items():
                        stable_label = track["stable_label"]
                        if stable_label == "unknown":
                            continue
                        if (now_mono - track["stable_since"]) < args.send_delay_seconds:
                            continue
                        if track["last_sent_label"] == stable_label:
                            continue

                        payload = {
                            "index": send_index,
                            "track_id": track_id,
                            "label": stable_label,
                            "class_index": class_to_idx.get(stable_label, None),
                            "confidence": round(float(track["conf"]), 4),
                            "timestamp_unix": time.time(),
                        }
                        if not send_tcp_message(tcp_sock, payload):
                            tcp_sock.close()
                            tcp_sock = None
                            break

                        track["last_sent_label"] = stable_label
                        track["last_sent_time"] = now_mono
                        send_index += 1

            for det in detections:
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                info = f"ID{det['track_id']} {det['stable_label']} ({det['stable_conf']:.2f})"
                cv2.putText(
                    frame,
                    info,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (50, 255, 50),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"area={int(det['area'])}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"objects={len(detections)} q=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Realtime Object Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
