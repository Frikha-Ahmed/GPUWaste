import numpy as np
import pandas as pd


# 1. Read the original GPU dataset from Ubuntu
path = "data/raw/data/utilization/kalos/GPU_UTIL.csv"

df = pd.read_csv(
    path,
    keep_default_na=False,
    na_values=[""],
    low_memory=False
)


# 2. Keep only GPU measurements (exclude Time)
measurements = df.drop(columns=["Time"])


# 3. Count missing measurements for each GPU
missing_count = measurements.isna().sum()


# 4. Convert measurements to numbers
numbers = measurements.apply(
    pd.to_numeric,
    errors="coerce"
)


# 5. Keep only valid utilization values (0 to 100)
valid_numbers = numbers.where(
    np.isfinite(numbers) &
    (numbers >= 0) &
    (numbers <= 100)
)


# 6. Calculate statistics for each GPU
valid_count = valid_numbers.count()

low_count = (valid_numbers < 10).sum()

invalid_count = (
    len(df) - valid_count - missing_count
)

avg_utilization = valid_numbers.mean().round(2)

low_pct = (100 * low_count / valid_count).round(2)


# 7. Create our final results table
result = pd.DataFrame({
    "gpu_id": valid_numbers.columns,
    "valid_count": valid_count.values,
    "low_count": low_count.values,
    "missing_count": missing_count.values,
    "invalid_count": invalid_count.values,
    "avg_utilization": avg_utilization.values,
    "low_utilization_pct": low_pct.values
})


# 8. Apply our GPUWaste detection rules
coverage = (
    100 * result["valid_count"] / len(df)
)

candidates = result[
    (result["avg_utilization"] < 10) &
    (result["low_utilization_pct"] >= 80) &
    (coverage >= 80) &
    (result["valid_count"] >= 1000)
]


# 9. Save results locally as Parquet
result.to_parquet(
    "benchmarks/pandas_gpu_summary.parquet",
    index=False
)


# 10. Display results
print("\n=== PANDAS GPU ANALYSIS ===")

print("Measurements timestamps:", len(df))
print("Monitored GPUs:", len(result))
print("Underutilization candidates:", len(candidates))

print("\nLowest-utilization GPUs:")

print(
    result.sort_values(
        "avg_utilization"
    ).head(15).to_string(index=False)
)
