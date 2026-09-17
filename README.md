# Telepati Agridata

Proyek ini dikembangkan untuk perlombaan **Telepati Agridata**. Repositori ini berisi notebook eksperimen, pemrosesan dataset, dan persiapan model *computer vision* serta *object detection*.

## Cara Menjalankan Proyek

### Prasyarat

Pastikan Python 3.12 atau lebih baru dan `uv` sudah terinstal.

Instal `uv` sesuai sistem operasi Anda:

**macOS/Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Instalasi

1. Clone repositori, lalu masuk ke direktorinya:

  ```bash
  git clone https://github.com/Isannali/lomba.git
  cd lomba
  ```

2. Sinkronkan environment dan dependensi:

  ```bash
  uv sync
  ```

3. Jalankan Jupyter Lab:

  ```bash
  uv run --with jupyter jupyter lab
  ```

### Jika Menggunakan VS Code

Buka repositori ini di VS Code, lalu pilih interpreter atau kernel Python dari environment `.venv` yang dibuat oleh `uv`.

### Pengaturan Dataset
1. Unduh dataset lomba terlebih dahulu ke komputer Anda.
2. Ekstrak file dataset tersebut ke direktori pilihan Anda.
3. Buka notebook eksperimen (.ipynb) di Jupyter Lab atau VS Code.
4. Cari variabel DATASET_ROOT dan YOLO_OUT di cell awal notebook, lalu ubah nilainya sesuai lokasi/path folder dataset sesuai dengan Path pada folder anda.
contoh:
```bash
DATASET_ROOT = "path/to/your/extracted_dataset"
YOLO_OUT="path/to/your/destination"
```



