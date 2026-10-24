# Telepati Agridata

Proyek ini menggunakan object detection untuk mengenali kondisi dan penyakit
tanaman padi. Pipeline eksperimen—mulai dari persiapan data, preprocessing,
pelatihan, evaluasi, hingga visualisasi prediksi—tersedia di
[lomba-full-pipeline.ipynb](./lomba-full-pipeline.ipynb).

## How to Run

Notebook dapat dijalankan di Kaggle maupun secara lokal (Windows, macOS, atau
Linux) menggunakan Jupyter Notebook/Lab. Untuk menjalankan lokal, clone atau
unduh repositori dan jalankan notebook dengan direktori kerja di root
repositori, yaitu direktori yang berisi notebook dan file `yolo26l_715.pt`.

```bash
git clone https://github.com/Isannali/lomba.git
cd lomba
```

### 1. Siapkan Python dan environment

Gunakan Python 3.10 atau lebih baru. Dari root repositori, buat dan aktifkan
virtual environment:

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependensi notebook dan Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install gdown matplotlib ultralytics pandas numpy pillow jupyter ipykernel
```

Jika ingin menggunakan GPU NVIDIA, pastikan driver CUDA pada komputer sesuai
dengan versi PyTorch yang digunakan. Ikuti petunjuk instalasi PyTorch untuk
memilih build CUDA yang cocok bagi sistem Anda. Notebook dapat dijalankan pada
CPU dengan mengganti `device=[0,1]` menjadi `device="cpu"`, tetapi training
YOLO26-L dapat berjalan sangat lambat. Untuk satu GPU, gunakan `device=0`;
sesuaikan `batch` jika memori GPU tidak mencukupi.

### 2. Sesuaikan path di notebook

Notebook awalnya memakai beberapa lokasi khusus Kaggle. Sebelum menjalankan
secara lokal, buka `lomba-full-pipeline.ipynb` dan sesuaikan **semua** path
berikut:

| Path di notebook | Ganti dengan |
| --- | --- |
| `/kaggle/working/...` | Path absolut ke root repositori lokal, misalnya `/home/nama/lomba/...` atau `C:/Users/Nama/lomba/...` |
| `/kaggle/input/models/rajasatricahya/yolo26l-715/pytorch/default/1/yolo26l_715.pt` | Path weights lokal, misalnya `yolo26l_715.pt` jika notebook dijalankan dari root repositori |

Path `/kaggle/working` digunakan pada cell preprocessing, pembuatan
`dataset.yaml`, training, dan evaluasi. Ganti bagian prefix-nya secara
konsisten; contoh, `/kaggle/working/Dataset_Telepati` menjadi
`/home/nama/lomba/Dataset_Telepati`. Cell-cell yang memakai path relatif seperti
`dataset.zip`, `dataset_raw/`, dan `Dataset_Telepati/` mengasumsikan direktori
kerja notebook adalah root repositori. Jangan ubah hanya beberapa path karena
cell berikutnya bergantung pada lokasi keluaran cell sebelumnya.

Satu cell pemeriksaan label juga berisi nama file contoh yang spesifik pada
dataset. Jika file tersebut tidak ditemukan pada dataset Anda, ubah
`label_file` di cell itu ke file `.txt` yang benar di
`Dataset_Telepati/labels/train/`, atau lewati cell pemeriksaan tersebut.

### 3. Jalankan notebook

Pastikan virtual environment masih aktif, lalu mulai Jupyter dari root
repositori:

```bash
jupyter lab
```

Buka `lomba-full-pipeline.ipynb`, pilih kernel dari virtual environment `.venv`,
dan jalankan cell dari atas ke bawah (`Run All`). Notebook akan mengunduh
`dataset.zip` dari Google Drive, mengekstraknya ke `dataset_raw/`, memproses
anotasi, membuat struktur dataset YOLO di `Dataset_Telepati/`, lalu melatih dan
mengevaluasi model. Jika sistem tidak menyediakan perintah `unzip` (misalnya
Windows tanpa utilitas tersebut), unduh dataset secara manual, ekstrak ke
`dataset_raw/`, lalu lewati cell unduh dan ekstrak (`gdown` dan `unzip`) dan
mulai dari cell preprocessing.

Weights yang disertakan di repositori digunakan oleh cell inferensi. Sementara
itu, cell evaluasi menggunakan `runs/detect/train/weights/best.pt`, yaitu
weights yang baru dibuat oleh proses training. Jalankan cell training terlebih
dahulu sebelum evaluasi; untuk hanya melakukan inferensi, pastikan cell
inferensi menunjuk ke file weights lokal dan dataset test sudah tersedia.

## Dependencies

Notebook memerlukan paket-paket berikut:

```bash
python -m pip install gdown matplotlib ultralytics pandas numpy pillow jupyter ipykernel
```

Dependensi yang digunakan:

- `ultralytics` — training, evaluasi, konversi anotasi COCO, dan inferensi YOLO.
- `gdown` — mengunduh arsip dataset dari Google Drive.
- `matplotlib` — visualisasi distribusi kelas dan hasil prediksi.
- `pandas`, `numpy`, `pillow` — pengolahan data dan gambar.
- `opencv-python` dan `PyYAML` — pemrosesan gambar serta pembacaan konfigurasi
  dataset; keduanya digunakan oleh notebook dan tersedia sebagai dependensi
  Ultralytics.
- Python standard library (`json`, `pathlib`, `shutil`, dan `collections`).

Training memerlukan PyTorch. Ultralytics memasang dependensi PyTorch yang
diperlukan; untuk GPU NVIDIA, gunakan instalasi PyTorch dengan dukungan CUDA
yang sesuai dengan driver dan perangkat keras lokal.

## Arsitektur Model

- **Model:** Ultralytics YOLO26-L, dibuat dari konfigurasi `yolo26l.yaml`.
- **Tugas:** deteksi objek dengan bounding box.
- **Inisialisasi training:** `pretrained=False`; model training dimulai dari
  konfigurasi, bukan dari weights pralatih.
- **Jumlah kelas:** 11 kelas penyakit/kondisi padi:
  `bacterial_leaf_blight`, `bacterial_panicle_blight`, `blast`, `brown_spot`,
  `false_smut`, `healthy`, `leaf_roller`, `leaf_scald`, `narrow_brown`,
  `sheath_blight`, dan `tungro`.

## Hyperparameter

Nilai berikut mengikuti pemanggilan `model.train()` di notebook:

| Parameter | Nilai |
| --- | --- |
| `epochs` | `300` |
| `imgsz` | `640` |
| `batch` | `32` |
| `device` | `[0, 1]` (ubah sesuai perangkat lokal) |
| `pretrained` | `False` |
| `hsv_h` | `0.01` |
| `hsv_s` | `0.4` |
| `hsv_v` | `0.4` |
| `erasing` | `0.2` |
| `fliplr` | `0.3` |
| `flipud` | `0.0` |
| `mosaic` | `0.7` |
| `patience` | `30` |

Training menggunakan early stopping dengan `patience=30`. Evaluasi dijalankan
pada split `test` dengan ukuran gambar `640`.

## Model Weights

- **Weights yang disertakan:** [yolo26l_715.pt](./yolo26l_715.pt)
- **Sumber model di Kaggle:** [yolo26l-715](https://www.kaggle.com/models/rajasatricahya/yolo26l-715)
- **Weights hasil training notebook:** `runs/detect/train/weights/best.pt`.
  File ini dihasilkan setelah training dan berbeda dari `yolo26l_715.pt`.
