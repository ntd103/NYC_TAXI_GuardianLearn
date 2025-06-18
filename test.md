run in container bash
```
jupyter kernelspec list
#    python3      /opt/conda/share/jupyter/kernels/python3
# (python3 kernel chính là nơi PySpark có mặt)
python - <<'PY'
import pyspark, sys
print("PySpark version:", pyspark.__version__)
PY
```