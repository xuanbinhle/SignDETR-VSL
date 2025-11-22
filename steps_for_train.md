<Hướng dẫn train mô hình>

1. git clone repo này

2. tạo .venv

3. activate venv

4. pip install uv

5. tạo thêm folder checkpoints và combineddata

6. unzip combineddata.zip rồi lưu folder train, val vô folder combineddata

7. chạy lệnh sau để train: uv run src\train.py

điều chỉnh batch size trong file train.py nếu vượt quá vram