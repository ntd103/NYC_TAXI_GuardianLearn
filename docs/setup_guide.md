# Hướng dẫn thiết lập môi trường phát triển

**Đối tượng**: Sinh viên, bạn muốn phát triển ứng dụng dự án `Dự đoán thời gian di chuyển của taxi ...` trên Windows 11, sử dụng Docker (WSL2) và code qua Visual Studio Code.

---

## Mục lục

1. [Cấu trúc thư mục dự án](#thu-muc)
2. [Cài đặt Docker và cấu hình all-spark-notebook](#docker)

   * 2.1. Kiểm tra Docker (trước khi cài)
   * 2.2. Gỡ cài đặt Docker (nếu đã cài trước)
   * 2.3. Cài đặt Docker Desktop trên Windows 11
   * 2.4. Cấu hình WSL2 và tích hợp Docker
   * 2.5. Thiết lập và chạy container all-spark-notebook
3. [Cấu hình Visual Studio Code](#vscode)

   * 3.1. Cài đặt VS Code
   * 3.2. Cài đặt các extension cần thiết
   * 3.3. Thiết lập Workspace và kết nối Docker Container
4. [Kiểm tra sau khi cài đặt](#kiem-tra)
5. [Hướng dẫn gỡ bỏ cài đặt](#go-bo)
6. [Xử lý lỗi thường gặp](#loi)

---

## 1. Cấu trúc thư mục dự án&#x20;

Tạo một thư mục gốc cho dự án, ví dụ `nyc_taxi_project/`, với cấu trúc như sau:

```
nyc_taxi_project/
├── data/                     # Dữ liệu thô (CSV, parquet)
│   ├── raw/
│   └── processed/
├── notebooks/                # Notebook thử nghiệm (Jupyter)
├── src/                      # Mã nguồn chính (PySpark, app)
│   ├── etl/
│   ├── train/
│   └── gui/
├── docker/                   # Dockerfile và cấu hình
│   └── all-spark-notebook/
│       ├── Dockerfile    # file tên gốc, D hoa f thường, không có phần mở rộng (extension)
│       └── environment.yml   # (nếu cần)
├── .devcontainer/            # VS Code Remote Container config
├── .vscode/                  # VS Code workspace settings
│   └── settings.json
├── README.md                 # Giới thiệu và hướng dẫn nhanh
└── setup_guide.md            # Hướng dẫn này
```

> **Lưu ý**: Tất cả file cấu hình và script đều nằm trong thư mục dự án để dễ backup và chia sẻ.

---

## 2. Cài đặt Docker và cấu hình all-spark-notebook&#x20;

### 2.1. Kiểm tra Docker (trước khi cài)

1. Mở PowerShell hoặc Command Prompt với quyền Administrator.
2. Chạy lệnh:

   ```bash
   docker --version
   ```
3. Nếu nhận về phiên bản Docker (ví dụ `Docker version 20.10.x, build ...`), Docker đã được cài sẵn.
4. Nếu dòng lệnh báo `docker: command not found` hoặc tương tự, tiếp tục cài đặt ở bước 2.3.

> **Bước yêu cầu phản hồi**: Nếu bạn đã có Docker, hãy cho tôi biết phiên bản; nếu chưa có, báo `Chưa cài`. Sau đó tôi sẽ hướng dẫn gỡ cài hoặc cài mới.

---

### 2.2. Gỡ cài đặt Docker (nếu đã cài trước)

> **CHỈ thực hiện nếu ứng dụng Docker phiền bản cũ hoặc cần tải lại**

1. Mở **Settings** → **Apps** → **Apps & features**.
2. Tìm `Docker Desktop`, chọn **Uninstall**.
3. Xóa các thư mục cấu hình cũ (nếu có):

   ```powershell
   Remove-Item -Recurse -Force "$Env:UserProfile\AppData\Roaming\Docker"
   Remove-Item -Recurse -Force "$Env:UserProfile\AppData\Roaming\docker-desktop"
   ```
4. Khởi động lại Windows.

---

### 2.3. Cài đặt Docker Desktop trên Windows 11

1. Truy cập trang tải Docker Desktop: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Tải bản dành cho **Windows (WSL 2)**.
3. Chạy installer và làm theo hướng dẫn.
4. Khi cài, chọn tích hợp WSL2.
5. Sau khi cài xong, mở Docker Desktop; đợi biểu tượng Docker chuyển sang trạng thái chạy.

### 2.4. Cấu hình WSL2 và tích hợp Docker

1. Bật tính năng WSL2:

   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
2. Tải Linux Kernel mới nhất: [https://aka.ms/wslkernel](https://aka.ms/wslkernel)
3. Thiết lập WSL2 làm mặc định:

   ```powershell
   wsl --set-default-version 2
   ```
4. Cài đặt bản phân phối Linux (Ubuntu) từ Microsoft Store.
5. Kiểm tra:

   ```bash
   wsl -l -v
   ```

   Đảm bảo Ubuntu có phiên bản WSL2.
6. Trong Docker Desktop ⇒ **Settings** ⇒ **Resources** ⇒ **WSL Integration**, bật tích hợp cho Ubuntu.

#### 2.5. Thiết lập và chạy container all-spark-notebook

> **Lưu ý**: Chạy lệnh trên máy chủ (host) hoặc WSL2 shell, không trong container.

1. Tạo file `docker/all-spark-notebook/Dockerfile`:

   ```dockerfile
   FROM jupyter/all-spark-notebook:latest
   USER root
   RUN apt-get update && apt-get install -y \
       vim \
       git
   USER $NB_UID
   ```

2. Xây image:

   ```bash
   cd docker/all-spark-notebook
   docker build -t all-spark-notebook:latest .
   ```

> **Khắc phục lỗi**: Nếu báo `failed to read dockerfile`, kiểm tra:
>
> * Đúng thư mục chứa `Dockerfile`.
> * Tên file là `Dockerfile` (chữ D viết hoa, f thường).
> * Terminal đang ở host, không phải container.

3. Chạy container:

   ```bash
   docker run -d \
     --name nyc-taxi-notebook \
     -p 8888:8888 \
     -v "${PWD}/data:/home/jovyan/data" \
     -v "${PWD}/notebooks:/home/jovyan/notebooks" \
     all-spark-notebook:latest
   ```

> **Tùy chỉnh tên container**: Bạn có thể đặt tên cho container thông qua tham số `--name`. Ví dụ, ở đây sử dụng `--name nyc-taxi-notebook`; nếu muốn đổi tên, chỉ cần thay `nyc-taxi-notebook` thành tên bạn chọn.

4. Mở trình duyệt tới `http://localhost:8888`, và đăng nhập token hiển thị ở log của container.

---

## 3. Cấu hình Visual Studio Code&#x20;

### 3.1. Cài đặt VS Code

1. Tải và cài VS Code cho Windows: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
2. Mở VS Code lần đầu, cấu hình ngôn ngữ nếu cần.

### 3.2. Cài đặt các Extension cần thiết

* **Dev Containers**: Kết nối và phát triển trực tiếp trong container Docker.
* **Python**: Hỗ trợ Python, Jupyter.
* **PySpark** (nếu có): Hỗ trợ cú pháp Spark.
* **GitLens**: Quản lý Git.
* **Docker**: Quản lý container, image.

### 3.3. Thiết lập Workspace và kết nối Docker Container

1. Tạo file `.devcontainer/devcontainer.json`:

   ```json
   {
     "name": "NYC Taxi Notebook",
     "context": "..",
     "dockerFile": "./docker/all-spark-notebook/Dockerfile",
     "workspaceFolder": "/home/jovyan",
     "extensions": [],
     "forwardPorts": [8888],
     "mounts": [
       "source=${localWorkspaceFolder}/data,target=/home/jovyan/data,type=bind",
       "source=${localWorkspaceFolder}/notebooks,target=/home/jovyan/notebooks,type=bind"
     ]
   }
   ```

2. Mở VS Code, chọn **Dev Containers: Open Folder in Container...**, trỏ đến thư mục dự án.

3. VS Code sẽ build và attach vào container. Mọi thao tác code, debug, Git sẽ thực hiện trong container.

---

## 4. Kiểm tra sau khi cài đặt&#x20;

1. Trong VS Code terminal (đã kết nối container), chạy:

   ```bash
   python --version
   spark-submit --version
   ```
2. Mở Jupyter Notebook, tạo notebook mới, thử import PySpark:

   ```python
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.getOrCreate()
   spark.range(5).show()
   ```
3. Nếu không lỗi, môi trường đã được thiết lập thành công.

---

## 5. Hướng dẫn gỡ bỏ cài đặt&#x20;

1. Trong VS Code, tắt container và xóa: **Dev Containers: Close Remote Connection**, sau đó:

   ```bash
   docker stop nyc-taxi-notebook && docker rm nyc-taxi-notebook
   docker image rm all-spark-notebook:latest
   ```
2. Gỡ VS Code extensions nếu cần: **View → Extensions** → tìm extension → Uninstall.
3. Gỡ Docker Desktop: như mục 2.2.

---

## 6. Đóng và mở lại dự án sau khi khởi động lại máy

Khi bạn tạm dừng công việc hoặc tắt máy, thực hiện các bước sau để đóng và sau đó mở lại toàn bộ môi trường dự án:

1. **Đóng kết nối Dev Container**:

   * Trong VS Code, nhấn biểu tượng `><` ở góc trái dưới cùng (trái của thanh status) và chọn **Dev Containers: Close Remote Connection**.
   * Chờ VS Code ngắt kết nối và trở về giao diện làm việc local.

2. **Dừng (stop) container** (nếu chỉ tạm dừng công việc):

   ```bash
   docker stop nyc-taxi-notebook
   ```

> *Tuỳ chọn:* Nếu muốn giải phóng tài nguyên hoàn toàn (xóa container), có thể thực hiện thêm:
>
> ```bash
> docker rm nyc-taxi-notebook
> ```

```bash
   docker stop nyc-taxi-notebook
   docker rm nyc-taxi-notebook
```

3. **Tắt Docker Desktop** nếu cần nghỉ ngơi lâu:

   * Mở Docker Desktop và chọn **Quit Docker Desktop**, hoặc
   * Trên Windows, mở Notification Area (góc phải taskbar), click phải vào icon Docker và chọn **Quit Docker Desktop**.

4. **Khởi động lại Docker và container** khi làm tiếp:

   * Mở Docker Desktop.
   * Mở terminal host, chuyển đến thư mục dự án và chạy lại:

     ```bash
     cd D:/NYC_Taxi_Project/docker/all-spark-notebook
     docker build -t all-spark-notebook:latest .  # nếu chưa có image hoặc đã rebuild
     docker run -d \
       --name nyc-taxi-notebook \
       -p 8888:8888 \
       -v "${PWD}/data:/home/jovyan/data" \
       -v "${PWD}/notebooks:/home/jovyan/notebooks" \
       all-spark-notebook:latest
     ```

5. **Mở lại trong VS Code**:

   * Mở VS Code tại thư mục dự án.
   * Chọn **Dev Containers: Open Folder in Container...** và chọn folder dự án.
   * VS Code sẽ tự động build/attach container như cấu hình trong `.devcontainer/devcontainer.json`.

6. **Kiểm tra**:

   * Trong terminal của VS Code (đang ở container), chạy `python --version` hoặc `spark-submit --version` để xác nhận môi trường sẵn sàng.

---

## 7. Xử lý lỗi thường gặp&#x20;

* **Lỗi kết nối WSL**: Chạy `wsl --update` và khởi động lại.
* **Port 8888 bị chiếm**: Thay `-p 8889:8888` trong lệnh `docker run`.
* **VS Code không attach Container**: Kiểm tra phiên bản `Remote - Containers` và Docker Engine đang chạy.

*Hướng dẫn chi tiết kết thúc. Vui lòng làm theo từng bước và báo kết quả kiểm tra khi hoàn thành từng mục để tôi có thể hỗ trợ tiếp.*
