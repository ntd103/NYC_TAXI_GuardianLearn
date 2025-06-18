# %% [markdown]
# ## Notebook 02 – Linear Regression (Spark)  
# **Mục tiêu**  
# 1. Xây mô hình hồi quy tuyến tính trên dữ liệu NYC Taxi Trip Duration 2016 (đã EDA & làm sạch).  
# 2. So sánh 3 phiên bản:  
#    * **Baseline LR** – đặc trưng gốc.  
#    * **+ Interaction** – thêm `distance_peak = distance_km × is_peak`.  
#    * **Optimized LR** – KMeans 10 cụm cho vị trí đón + Polynomial degree 2 cho `distance_km` + Elastic-Net (Ridge/Lasso) tìm tham số bằng Cross-Validation.  
# 3. Đánh giá bằng RMSLE và RMSE (giây thực), chọn mô hình tốt nhất.  
# 4. Lưu model
# 

# %%
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, PolynomialExpansion
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import math

spark = SparkSession.builder.appName("TaxiLR").getOrCreate()
DATA_PATH = "/home/jovyan/work/data/processed/cleaned_data.csv"

df = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH))

print("Rows:", df.count(), "  Cols:", len(df.columns))


# %% [markdown]
# ### Đặc trưng gốc sử dụng
# 
# Trước khi tiến hành xây dựng pipeline, chúng ta phân loại các feature theo 3 nhóm cơ bản:
# 
# | Nhóm          | Cột                                                |
# |---------------|----------------------------------------------------|
# | Numeric       | `distance_km`, `pickup_hour`, `passenger_count`    |
# | Binary        | `is_peak`, `airport_trip`, `store_and_fwd_flag_num`|
# | Categorical   | `vendor_id`, `pickup_weekday`, `pickup_month`      |
# 
# - **pickup_weekday** và **pickup_month** được bổ sung để khai thác thông tin tuần và tháng, nhằm bắt xu hướng theo thời gian.
# - **Nhãn** mục tiêu là `log_duration` (đã được log1p transform) giúp ổn định phương sai và phù hợp với RMSE trên log.
# 
# 

# %% [markdown]
# ## Định nghĩa hàm đánh giá mô hình
# 
# Hàm `eval_metrics` tính toán ba chỉ số:
# 1. **RMSE_log**: sai số gốc trên không gian log (tương đương RMSLE).  
# 2. **RMSE_sec**: sai số trên đơn vị giây gốc (bằng cách expm1 kết quả).  
# 3. **RMSLE**: root mean squared logarithmic error, thể hiện tương quan giữa sai số tương đối.

# %%
def eval_metrics(pred_df):
    # RMSE trên log (≈ RMSLE)
    ev_log = RegressionEvaluator(labelCol="log_duration",
                                 predictionCol="prediction",
                                 metricName="rmse")
    rmse_log = ev_log.evaluate(pred_df)

    exp_df = (pred_df
              .withColumn("pred_sec",  F.expm1("prediction"))
              .withColumn("label_sec", F.expm1("log_duration")))
    # RMSE trên giây gốc
    rmse_sec = RegressionEvaluator(labelCol="label_sec",
                                   predictionCol="pred_sec",
                                   metricName="rmse").evaluate(exp_df)
    # RMSLE
    rmsle = (exp_df
             .select(F.pow(F.log1p("pred_sec") - F.log1p("label_sec"), 2).alias("sq"))
             .agg(F.sqrt(F.mean("sq")).alias("rmsle"))
             .first()["rmsle"])
    return rmse_log, rmse_sec, rmsle


# %% [markdown]
# Baseline Linear Regression
# 
# Pipeline baseline gồm:
# 1. **StringIndexer** cho `vendor_id` chuyển thành `vendor_idx`.  
# 2. **VectorAssembler** gom các feature raw thành vector.  
# 3. **StandardScaler** chuẩn hóa về phân phối chuẩn.  
# 4. **LinearRegression** trên target `log_duration`, với `maxIter=50`, `regParam=0.1`.
# 
# Feature set gồm cả `pickup_weekday` và `pickup_month` để đánh giá tác động bổ sung.

# %%
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# StringIndexer vendor
vendor_idx = StringIndexer(inputCol="vendor_id",
                           outputCol="vendor_idx",
                           handleInvalid="keep")

features_base = ["distance_km", "pickup_hour", "passenger_count",
                 "is_peak", "airport_trip", "store_and_fwd_flag_num",
                 "vendor_idx"]

assembler = VectorAssembler(inputCols=features_base,
                            outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw",
                        outputCol="features",
                        withMean=True, withStd=True)

lr = LinearRegression(featuresCol="features",
                      labelCol="log_duration",
                      maxIter=50, regParam=0.1)

pipe_base = Pipeline(stages=[vendor_idx, assembler, scaler, lr])
model_base = pipe_base.fit(train_df)
pred_base  = model_base.transform(test_df)

rmse_log_b, rmse_sec_b, rmsle_b = eval_metrics(pred_base)
print(f"Baseline → RMSE_sec={rmse_sec_b:.2f}s , RMSLE={rmsle_b:.4f}")


# %%
# Baseline với full feature set (có pickup_weekday & pickup_month)
# Train–Test split
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# StringIndexer vendor
vendor_idx = StringIndexer(inputCol="vendor_id",
                           outputCol="vendor_idx",
                           handleInvalid="keep")

# Features bao gồm giờ, ngày trong tuần, tháng
features_base = [
    "distance_km", "pickup_hour", "passenger_count",
    "is_peak", "airport_trip", "store_and_fwd_flag_num",
    "pickup_weekday", "pickup_month",
    "vendor_idx"
]

assembler = VectorAssembler(inputCols=features_base,
                            outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw",
                        outputCol="features",
                        withMean=True, withStd=True)

lr = LinearRegression(featuresCol="features",
                      labelCol="log_duration",
                      maxIter=50, regParam=0.1)

pipe_base = Pipeline(stages=[vendor_idx, assembler, scaler, lr])
model_base = pipe_base.fit(train_df)
pred_base  = model_base.transform(test_df)

rmse_log_b, rmse_sec_b, rmsle_b = eval_metrics(pred_base)
print(f"Baseline → RMSE_sec={rmse_sec_b:.2f}s , RMSLE={rmsle_b:.4f}")


# %% [markdown]
# Chúng tôi tạo thêm feature `distance_peak = distance_km × is_peak` để mô hình học tương tác giữa khoảng cách và giờ cao điểm.
# 

# %%
# Thêm interaction distance_peak
df_int = df.withColumn("distance_peak",
                       F.col("distance_km") * F.col("is_peak"))

train_i, test_i = df_int.randomSplit([0.7, 0.3], seed=42)

features_int = features_base + ["distance_peak"]
assembler_i  = VectorAssembler(inputCols=features_int,
                               outputCol="features_raw")
pipe_int = Pipeline(stages=[vendor_idx, assembler_i, scaler, lr])

model_int = pipe_int.fit(train_i)
pred_int  = model_int.transform(test_i)

rmse_log_i, rmse_sec_i, rmsle_i = eval_metrics(pred_int)
print(f"+Interaction → RMSE_sec={rmse_sec_i:.2f}s , RMSLE={rmsle_i:.4f}")


# %% [markdown]
# KMeans, Polynomial & Cross-Validation
# 
# Để tối ưu hoá, chúng tôi thêm:
# 1. **KMeans (k=10)** phân nhóm điểm đón (`pickup_cluster`).  
# 2. Index hóa cluster thành `cluster_idx`.  
# 3. Tạo `PolynomialExpansion(degree=2)` trên vector chuẩn hoá.  
# 4. **CrossValidator** với grid search trên `regParam` và `elasticNetParam` của Linear Regression.
# 

# %%
# KMeans + Polynomial + CV Tuning (Optimized)
# 1) Cluster vị trí đón
vec_pos = VectorAssembler(
    inputCols=["pickup_longitude","pickup_latitude"],
    outputCol="pos_vec")
df_pos = vec_pos.transform(df)

km = KMeans(k=10, seed=42,
            featuresCol="pos_vec",
            predictionCol="pickup_cluster")
df_km = km.fit(df_pos).transform(df_pos)

cluster_idx = StringIndexer(inputCol="pickup_cluster",
                            outputCol="cluster_idx",
                            handleInvalid="keep")
df_km = cluster_idx.fit(df_km).transform(df_km)

# 2) Assemble + scale + poly
features_km = features_base + ["cluster_idx"]
assembler_km = VectorAssembler(inputCols=features_km,
                               outputCol="features_raw")
scaler_km = StandardScaler(inputCol="features_raw",
                           outputCol="feat_scaled",
                           withMean=True, withStd=True)
poly = PolynomialExpansion(degree=2,
                           inputCol="feat_scaled",
                           outputCol="features")

lr_opt = LinearRegression(featuresCol="features",
                          labelCol="log_duration",
                          maxIter=100)

pipe_opt = Pipeline(stages=[
    vendor_idx,
    assembler_km,
    scaler_km,
    poly,
    lr_opt
])

evaluator = RegressionEvaluator(labelCol="log_duration",
                                predictionCol="prediction",
                                metricName="rmse")  # metric trên log

paramGrid = (ParamGridBuilder()
             .addGrid(lr_opt.regParam,        [0.01, 0.05, 0.1])
             .addGrid(lr_opt.elasticNetParam, [0.0, 0.3, 0.7])
             .build())

cv = CrossValidator(estimator=pipe_opt,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3,
                    parallelism=2)

train_opt, test_opt = df_km.randomSplit([0.7,0.3], seed=42)
cvModel = cv.fit(train_opt)
best = cvModel.bestModel
pred_opt = best.transform(test_opt)

rmse_log_o, rmse_sec_o, rmsle_o = eval_metrics(pred_opt)
print(f"Optimized → RMSE_sec={rmse_sec_o:.2f}s , RMSLE={rmsle_o:.4f}")


# %% [markdown]
# Tổng hợp kết quả ba phiên bản mô hình vào bảng để tiện so sánh.
# 

# %%
# Tổng hợp kết quả
import pandas as pd

results = pd.DataFrame({
    "Model": ["Baseline", "+Interaction", "Optimized"],
    "RMSLE": [rmsle_b, rmsle_i, rmsle_o],
    "RMSE_sec": [rmse_sec_b, rmse_sec_i, rmse_sec_o]
})
display(results)



