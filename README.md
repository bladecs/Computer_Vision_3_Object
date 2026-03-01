# Computer Vision Starter (Python)

## 1) Buat virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install dependensi umum
```powershell
pip install -r requirements.txt
```

## 3) Install PyTorch (pilih salah satu)
CPU:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 4) Jalankan baseline training
```powershell
python src/train.py
```

## Struktur proyek
- `data/` dataset mentah / split
- `notebooks/` eksperimen
- `src/train.py` baseline training
- `src/models/` arsitektur model custom
- `src/utils/` helper dataset, metrics, dll

## 5) Jalankan realtime testing
```powersheel
python src/test_realtime.py --checkpoint checkpoints/custom_cnn_blue.pth --camera-index 0 --min-area 2200 --conf-threshold 0.65 --vote-window 10 --min-votes 6 --small-max-area 7500 --medium-max-area 55000
```
