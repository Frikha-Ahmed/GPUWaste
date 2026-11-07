import numpy as np
import pandas as pd


# 1. Dataset location
path = "data/raw/data/utilization/seren/GPU_UTIL.csv"

# Read the GPU identifiers without loading the data
gpu_ids = pd.read_csv(path, nrows=0).columns[1:]


# 2. Initialize our counters
n = len(gpu_ids)

valid_count = np.zeros(n, dtype=int)
missing_count = np.zeros(n, dtype=int)
low_count = np.zeros(n, dtype=int)
utilization_sum = np.zeros(n)

total_rows = 0


# 3. Process the CSV in chunks of 1000 rows
for chunk in pd.read_csv(
    path,
    chunksize=1000,
    dtype=str,
    keep_default_na=False
):

    # Ignore the timestamp column
    raw = chunk.iloc[:, 1:]

    # Count empty measurements
    missing_count += raw.eq("").sum(axis=0).to_numpy()

    # Convert values to numbers
    values = raw.apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    # A valid utilization is finite and between 0 and 100
    valid = (
        np.isfinite(values)
        & (values >= 0)
        & (values <= 100)
    )

    # Update statistics for each GPU
    valid_count += valid.sum(axis=0)

    low_count += (valid & (values < 10)).sum(axis=0)

    utilization_sum += np.where(
        valid, values, 0
    ).sum(axis=0)

    total_rows += len(chunk)


# 4. Calculate final statistics
invalid_count = total_rows - valid_count - missing_count

# Avoid division by zero when a GPU has no valid measurements
avg = np.divide(
    utilization_sum,
    valid_count,
    out=np.full(n, np.nan),
    where=valid_count > 0
)

low_pct = np.divide(
    100 * low_count,
    valid_count,
    out=np.full(n, np.nan),
    where=valid_count > 0
)


# 5. Create our result table
result = pd.DataFrame({
    "gpu_id": gpu_ids,
    "valid_count": valid_count,
    "low_count": low_count,
    "missing_count": missing_count,
    "invalid_count": invalid_count,
    "avg_utilization": np.round(avg, 2),
    "low_utilization_pct": np.round(low_pct, 2)
})


# 6. Apply the same detection rules as GPUWaste
coverage = 100 * result["valid_count"] / total_rows

candidates = result[
    (result["avg_utilization"] < 10)
    & (result["low_utilization_pct"] >= 80)
    & (coverage.round(2) >= 80)
    & (result["valid_count"] >= 1000)
]


# 7. Save our results locally
result.to_parquet(
    "benchmarks/seren_pandas.parquet",
    index=False
)


# 8. Display results
print("\n=== SEREN PANDAS BENCHMARK ===")
print("Timestamps:", total_rows)
print("GPUs:", len(result))
print("Potentially underutilized GPUs:", len(candidates))

print(result.sort_values("avg_utilization").head(15))
