# Project: Nhận diện Hành vi Bất thường (UCF-Crime) - Core Model

## Giới thiệu 🎯

Dự án này tập trung vào việc xây dựng một mô hình Deep Learning **"hạt nhân" (core model)** để tự động phát hiện và phân loại các hành vi bất thường trong video giám sát, sử dụng bộ dữ liệu **UCF-Crime**. Mô hình này sử dụng kiến trúc kết hợp CNN và LSTM làm nền tảng cho việc nhận diện hành vi trong môi trường giám sát thực tế.

---

## Dataset: UCF-Crime 🎬

* **Nguồn:** [UCF Center for Research in Computer Vision - Real-world Anomaly Detection](https://www.crcv.ucf.edu/projects/real-world/)
* **Đặc điểm:** Bao gồm video dài từ camera giám sát, chứa 13 loại hành vi bất thường (`Abuse`, `Arrest`, `Arson`, `Assault`, `Burglary`, `Explosion`, `Fighting`, `RoadAccidents`, `Robbery`, `Shooting`, `Shoplifting`, `Stealing`, `Vandalism`) và video bình thường (`Normal`).
* **Cách sử dụng:**
    * **Tập Huấn luyện (Train):** Sử dụng các video được gán **nhãn yếu (Weakly Labeled)** từ file `Action_Regnition_splits/train_001.txt`. Các video gốc này được **cắt thành các clip 5 giây** (sử dụng cửa sổ trượt 2 giây) để **tăng cường dữ liệu (Data Augmentation)** và chống học vẹt (Overfitting). Mặc dù được cắt nhỏ, các clip này vẫn giữ nguyên nhãn yếu của video gốc.
    * **Tập Kiểm thử (Test):** Sử dụng các video được gán **nhãn mạnh (Strongly Labeled)** từ file `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`. Dữ liệu này được **cắt thành các clip "sạch"** chỉ chứa hành vi tương ứng để đánh giá chính xác hiệu suất của mô hình.

---

## Kiến trúc Mô hình: CNN + LSTM 🧠

Mô hình hạt nhân sử dụng kiến trúc kết hợp:

1.  **CNN (Convolutional Neural Network):** Sử dụng **ResNet50** (pre-trained trên ImageNet, các lớp được đóng băng - frozen) làm bộ trích xuất đặc trưng không gian (spatial features) từ từng khung hình (frame).
2.  **LSTM (Long Short-Term Memory):** Nhận chuỗi các vector đặc trưng (20 frame/clip) từ CNN và học các mối quan hệ theo thời gian (temporal dependencies) để hiểu hành động.
3.  **Classification Layer:** Một lớp Fully Connected (với Dropout) để phân loại thành 1 trong 14 lớp.

*Tăng tốc độ huấn luyện được thực hiện bằng **Mixed Precision (AMP)**.*



---

## Cài đặt ⚙️

1.  Clone repository:
    ```bash
    git clone [URL-repository-cua-ban]
    cd [ten-repository]
    ```
2.  Tạo môi trường ảo (khuyến nghị):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate  # Windows
    ```
3.  Cài đặt thư viện:
    ```bash
    pip install torch torchvision opencv-python-headless pandas matplotlib scikit-learn tqdm
    # Hoặc pip install -r requirements.txt (nếu có file)
    ```

---

## Chuẩn bị Dữ liệu 💾

1.  **Tải Dataset:** Tải UCF-Crime và giải nén vào `DATA/RawVideo/` (cấu trúc thư mục con theo nhãn).
2.  **Tải File Split:** Tải `UCF_Crimes-Train-Test-Split.zip` và giải nén, đảm bảo có thư mục `Action_Regnition_splits`.
3.  **Chạy Script Chuẩn bị:**
    * **Tập Train:** Chạy `prepare_train_data_SLIDING_WINDOW.py`. Script này sẽ đọc `Action_Regnition_splits/train_001.txt`, tìm video trong `DATA/RawVideo/`, cắt thành các clip 5 giây (trượt 2 giây) và lưu vào `data_clips/train/`.
    * **Tập Test:**
        * Chạy script (hoặc code) để xử lý `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` thành file `cleaned_annotations.csv`.
        * Chạy `prepare_data.py`. Script này sẽ đọc `cleaned_annotations.csv`, tìm video trong `DATA/RawVideo/`, cắt thành các clip "sạch" theo frame và lưu vào `data_clips/test/`.

---

## Huấn luyện 🚀

* **Môi trường:** Khuyến nghị sử dụng **Kaggle Notebooks** (với **GPU T4**) để huấn luyện.
* **Thực thi:**
    * Nén thư mục `data_clips` thành `data_clips.zip`.
    * Tải `data_clips.zip` lên Kaggle Datasets.
    * Tải file Jupyter Notebook `PBL_Kaggle_Training.ipynb` lên Kaggle Code.
    * Trong Notebook Kaggle:
        * Bật GPU Accelerator (T4).
        * "Add Data" để kết nối Dataset đã tải lên.
        * Sửa lại các đường dẫn trong code (Cell 2, Cell 5) cho đúng với môi trường Kaggle.
        * Chọn **"Save Version" -> "Save & Run All (Commit)"** để chạy huấn luyện trong nền.
* **Checkpointing:**
    * `pbl_latest_checkpoint.pth`: Luôn lưu epoch mới nhất (ghi đè), dùng để tiếp tục huấn luyện.
    * `pbl_best_model.pth`: Chỉ lưu model có Test Accuracy cao nhất.
* **Kết quả:** Model tốt nhất được lưu vào `pbl_final_model.pth` (file này chỉ chứa state_dict), lịch sử huấn luyện (`loss`, `accuracy`) lưu vào `training_history.json`. Các file này sẽ nằm trong tab "Output" của phiên bản Kaggle đã chạy xong.

---

## Sử dụng (Dự đoán) 🔍

1.  Tải file model đã huấn luyện (`pbl_final_model.pth`) về máy.
2.  Sử dụng file `predict.py`:
    * Đảm bảo class `CnnRnn` được định nghĩa trong file.
    * Cập nhật `MODEL_PATH`, `NUM_CLASSES`, `SEQUENCE_LENGTH`, `CLASS_NAMES` cho đúng.
    * Chạy file với đường dẫn video cần dự đoán: `python predict.py --video_path /duong/dan/video_moi.mp4` (Bạn cần thêm argparse vào `predict.py`).
    * Script sẽ in ra nhãn dự đoán và độ tự tin.

---

## Kết quả & Phân tích 📈


---

## Cấu trúc File Dự án 📁
/PBL_Project/ ├── DATA/ │ └── RawVideo/ │ ├── Abuse/ │ ├── ... (13 thư mục lớp bất thường) │ └── Normal/ ├── data_clips/ │ ├── train/ # ~33k clips 5s (nhãn yếu) │ └── test/ # ~4.5k clips sạch (nhãn mạnh) ├── Action_Regnition_splits/ # File chia train/test gốc │ └── train_001.txt ├── .gitignore # Bỏ qua file .pth, .zip, venv... ├── PBL_Kaggle_Training.ipynb # Notebook huấn luyện (Kaggle/Colab) ├── prepare_data.py # Script cắt tập test ├── prepare_train_data_SLIDING_WINDOW.py # Script cắt tập train ├── predict.py # Script chạy dự đoán ├── requirements.txt # (Nên tạo) Danh sách thư viện └── README.md # File này
