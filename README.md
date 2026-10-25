# Telepati Agridata

## Cara menjalankan

Notebook [agridata_main.ipynb](./agridata_main.ipynb) dapat dijalankan di Kaggle
atau lingkungan Jupyter lokal. Jalankan semua sel secara berurutan dari sel
paling atas sampai akhir. Koneksi internet diperlukan agar notebook dapat
mengunduh dataset melalui `gdown`. Untuk training dengan waktu yang wajar,
gunakan GPU yang kompatibel dengan PyTorch CUDA. Eksperimen yang dilaporkan
menggunakan dua GPU dan memerlukan sekitar 9,2 jam untuk 132 epoch; waktu aktual
bergantung pada GPU dan lingkungan yang digunakan. Seed training ditetapkan ke
`42`.

Parameter di sel konfigurasi (sel 1 setelah sel instalasi):

- `TRAIN`: `True` menjalankan training; `False` melewati training dan
  menggunakan bobot yang tersedia.
- `EPOCHS`: batas maksimum epoch, yaitu `300`; early stopping menghentikan
  eksperimen yang dilaporkan pada epoch 132.
- `DEVICE`: notebook memilih seluruh GPU CUDA yang terdeteksi, misalnya `[0, 1]`
  pada eksperimen dua GPU atau `[0]` pada sistem satu GPU. Jika tidak ada GPU,
  notebook memilih CPU; training CPU dapat berjalan jauh lebih lambat.
- `WEIGHTS_OVERRIDE`: path opsional ke file bobot `.pt`. Saat `TRAIN=False`,
  notebook memeriksa path ini terlebih dahulu sebelum `BEST_WEIGHTS`.

### Evaluasi test atau inferensi dengan bobot rilis

Unduh [yolo26l_best_717.pt dari GitHub
Releases](https://github.com/Isannali/lomba/releases/tag/v1.1/yolo26l_best_717.pt)
ke komputer atau environment tempat notebook berjalan. Letakkan file di root
repositori, atau sesuaikan path di `WEIGHTS_OVERRIDE`. Contoh:

```python
TRAIN = False
WEIGHTS_OVERRIDE = "yolo26l_best_717.pt"
```

Path relatif dihitung dari direktori kerja notebook. Jalankan sel dari atas
sampai dan termasuk **Evaluasi** untuk menyiapkan data dan mengukur performa
bobot pada split test. Lanjutkan ke **Inferensi** untuk melihat prediksi pada
contoh gambar test.

Untuk prediksi pada gambar sendiri tanpa menjalankan evaluasi test, gunakan
Ultralytics dengan path bobot dan gambar yang sesuai:

```python
from ultralytics import YOLO

model = YOLO("yolo26l_best_717.pt")
results = model.predict(source="path/to/image.jpg", save=True)
```

Hasil prediksi disimpan Ultralytics di `runs/detect/predict/` (atau direktori
bernomor berikutnya jika direktori tersebut sudah ada).

## Dependensi

Versi runtime dan paket dari log eksperimen:

- Python `3.13.15`
- PyTorch `2.11.0+cu128`
- Ultralytics `8.4.187`
- `gdown`
- `matplotlib`
- `PyYAML` (`pyyaml`)
- `opencv-python` (`opencv`)

Sel instalasi notebook mengunci Ultralytics ke `ultralytics==8.4.187`.
PyTorch perlu dipasang dengan build yang sesuai dengan sistem operasi, driver,
dan GPU yang digunakan. Pada lingkungan yang mendukung file requirements
repositori, dependensi proyek juga tercantum di
[`requirements.txt`](./requirements.txt).

## Arsitektur Model

Model yang digunakan adalah **YOLO26l** (Ultralytics), detektor objek satu tahap, yang dibangun dari
`yolo26l.yaml` dan **dilatih dari nol tanpa bobot pretrained** (`pretrained=False`).

- Konfigurasi arsitektur: [`yolo26.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml)
  (ukuran `l` dipilih lewat faktor skala `[depth 1,0 ; width 1,0 ; max channel 512]`)
- Dokumentasi model: https://docs.ultralytics.com/models/yolo26/
- Rancangan: end-to-end (`end2end: True`), tanpa DFL (`reg_max: 1`), tiga skala keluaran P3/8, P4/16, P5/32
- Input 640 x 640, jumlah kelas 11 (kelas canonical lomba)
- Ukuran model (setelah layer digabung/fused, 11 kelas): **188 layer, 24.754.221 parameter, 86,5 GFLOPs**
- Versi library: Ultralytics 8.4.187, torch 2.11.0+cu128, Python 3.13.15

| No | Modul | Ulang | Kanal | Keterangan |
|---|---|---|---|---|
| 0 | Conv (k3, s2) | 1 | 64 | P1/2 |
| 1 | Conv (k3, s2) | 1 | 128 | P2/4 |
| 2 | C3k2 | 2 | 256 | |
| 3 | Conv (k3, s2) | 1 | 256 | P3/8 |
| 4 | C3k2 | 2 | 512 | |
| 5 | Conv (k3, s2) | 1 | 512 | P4/16 |
| 6 | C3k2 (c3k=True) | 2 | 512 | |
| 7 | Conv (k3, s2) | 1 | 512 | P5/32 |
| 8 | C3k2 (c3k=True) | 2 | 512 | |
| 9 | SPPF (k5) | 1 | 512 | |
| 10 | C2PSA | 2 | 512 | blok atensi (akhir backbone) |
| 11 | Upsample x2 | 1 | - | head |
| 12 | Concat dengan layer 6 | 1 | - | |
| 13 | C3k2 (c3k=True) | 2 | 512 | |
| 14 | Upsample x2 | 1 | - | |
| 15 | Concat dengan layer 4 | 1 | - | |
| 16 | C3k2 (c3k=True) | 2 | 256 | keluaran P3/8 (objek kecil) |
| 17 | Conv (k3, s2) | 1 | 256 | |
| 18 | Concat dengan layer 13 | 1 | - | |
| 19 | C3k2 (c3k=True) | 2 | 512 | keluaran P4/16 (objek sedang) |
| 20 | Conv (k3, s2) | 1 | 512 | |
| 21 | Concat dengan layer 10 | 1 | - | |
| 22 | C3k2 (c3k=True, e=0,5, attn=True) | 1 | 512 | keluaran P5/32 (objek besar) |
| 23 | Detect (layer 16, 19, 22) | 1 | - | 11 kelas |

## Hyperparameter

Nilai berikut digunakan pada sel training:

| Parameter | Nilai |
| --- | ---: |
| `epochs` | 300 (early stopping pada epoch 132; hasil terbaik epoch 102) |
| `imgsz` | 640 |
| `batch` | 32 |
| `device` | `[0, 1]` pada eksperimen |
| `patience` | 30 |
| `seed` | 42 |
| `hsv_h` | 0,01 |
| `hsv_s` | 0,4 |
| `hsv_v` | 0,4 |
| `erasing` | 0,2 |
| `fliplr` | 0,3 |
| `flipud` | 0,3 |
| `mosaic` | 0,7 |
| `pretrained` | `False` |

Parameter lain menggunakan nilai bawaan Ultralytics 8.4.187. Skor mandiri pada
split test: **mAP50 0,7166** dan **F1 0,7252** (precision 0,730; recall 0,721).
F1 dihitung sebagai rata-rata harmonik precision dan recall:
`2 × P × R / (P + R)`.

## Tautan bobot model

Bobot model: [`yolo26l_best_717.pt` (GitHub Release
v1.1)](https://github.com/Isannali/lomba/releases/tag/v1.1/yolo26l_best_717.pt).
Unduh file tersebut langsung ke lingkungan yang menjalankan notebook, lalu
atur `WEIGHTS_OVERRIDE` ke lokasi file seperti contoh pada bagian **Cara
menjalankan**.
