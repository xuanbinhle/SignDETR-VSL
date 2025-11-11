git clone repo này
tạo .venv
activate venv
pip install uv
tạo thêm folder checkpoints và combineddata
unzip combineddata.zip rồi lưu folder train, val vô folder combineddata
chạy lệnh sau để train: uv run src\train.py
điều chỉnh batch size trong file train.py nếu vượt quá vram