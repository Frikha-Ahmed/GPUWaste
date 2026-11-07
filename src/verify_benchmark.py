import pandas as pd
import numpy as np

# 1. Read both benchmark results
pandas_df = pd.read_parquet("benchmarks/seren_pandas.parquet")
spark_df = pd.read_parquet("benchmarks/seren_spark")

# 2. Sort both tables by GPU identifier
pandas_df = pandas_df.set_index("gpu_id").sort_index()
spark_df = spark_df.set_index("gpu_id").sort_index()

# 3. Verify that both contain the same GPUs
assert pandas_df.index.equals(spark_df.index)

print("GPU identifiers match:", len(pandas_df))

# 4. Compare integer statistics
columns = [
    "valid_count",
    "low_count",
    "missing_count",
    "invalid_count"
]

for column in columns:
    assert (pandas_df[column] == spark_df[column]).all()
    print(column, ": MATCH")

# 5. Compare calculated percentages and averages
columns = [
    "avg_utilization",
    "low_utilization_pct"
]

for column in columns:
    assert np.allclose(
        pandas_df[column],
        spark_df[column],
        atol=0.01,
        rtol=0,
        equal_nan=True
    )
    print(column, ": MATCH")

print("\nSUCCESS: Both implementations produce equivalent results!")
