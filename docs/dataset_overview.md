## Mô tả Bộ Dữ liệu (Dataset Overview)

**Nguồn:** [NYC Taxi Trip Duration – *Kaggle Competition Dataset*](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview). Bộ dữ liệu đã được làm sạch một phần, kích thước vừa phải (~1,4 triệu dòng trong tập huấn luyện), và có biến mục tiêu rõ ràng (`trip_duration`). 

**Các tệp chính:**

- **`train.csv`:** Tập dữ liệu huấn luyện, chứa đầy đủ các đặc trưng đầu vào và **biến mục tiêu** (`trip_duration`).
- **`test.csv`:** Tập dữ liệu kiểm thử, có cấu trúc giống `train.csv` nhưng **không chứa** cột `trip_duration`. Dùng để tự đánh giá mô hình sau khi huấn luyện.

**Các cột dữ liệu quan trọng:**

| Tên Cột             | Kiểu dữ liệu | Mô tả                                                  | Vai trò trong dự án                |
| ------------------- | ------------ | ------------------------------------------------------ | ---------------------------------- |
| `id`                | string       | ID định danh duy nhất cho mỗi chuyến đi                | Định danh (khóa chính cho bản ghi) |
| `vendor_id`         | integer      | ID của nhà cung cấp dịch vụ taxi (hãng taxi)           | Đặc trưng phân loại (categorical)  |
| `pickup_datetime`   | datetime     | Thời điểm bắt đầu chuyến đi                             | Đặc trưng thời gian (trích xuất giờ, ngày, v.v.) |
| `passenger_count`   | integer      | Số lượng hành khách trên xe                            | Đặc trưng dự đoán (numeric feature)|
| `pickup_longitude`  | float        | Kinh độ của điểm đón khách                              | Đặc trưng không gian (tính khoảng cách) |
| `pickup_latitude`   | float        | Vĩ độ của điểm đón khách                                | Đặc trưng không gian (tính khoảng cách) |
| `dropoff_longitude` | float        | Kinh độ của điểm trả khách                              | Đặc trưng không gian (tính khoảng cách) |
| `dropoff_latitude`  | float        | Vĩ độ của điểm trả khách                                | Đặc trưng không gian (tính khoảng cách) |
| `store_and_fwd_flag`| char         | Cờ lưu và chuyển tiếp dữ liệu của xe (Y hoặc N)         | Đặc trưng phân loại (nhị phân Y/N) |
| `trip_duration`     | integer      | **Biến mục tiêu** – Thời gian di chuyển (tính bằng giây) | Mục tiêu dự đoán của mô hình      |
