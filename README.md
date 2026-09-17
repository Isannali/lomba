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
  git clone https://github.com/USERNAME/NAMA-REPO.git
  cd NAMA-REPO
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


