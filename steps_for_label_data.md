<Hướng dẫn cho việc thu thập và gán nhãn dữ liệu>

0. Tạo môi trường: `python -m venv .venv` và trong terminal chạy các lệnh này: `pip install uv` `uv sync`
1. Activate môi trường: `.venv\Scripts\Activate.ps1`
2. Vào src\config.json để ghi các classes cần thu thập dữ liệu.
3. Thu thập dữ liệu bằng cách chạy script `uv run src/utils/collect_images.py` 
(số lượng ảnh cho mỗi lớp là 30, có thể điều chỉnh nếu thấy cần nhiều/ít hơn)
4. Gán nhãn dữ liệu: Chạy script `uv run label-studio`
Tạo project -> upload hình ảnh -> gán nhãn -> export dưới dạng 'Yolo with images'.
5. Lưu data về dưới dạng zip, upload lên drive.