import time
import numpy as np
import pandas as pd


# ==========================================
# PART 1 — GPU POWER ANALYSIS
# ==========================================

start = time.perf_counter()

path = "data/raw/data/utilization/seren/POWER_USAGE.csv"

# Get GPU names without loading the whole file
gpu_ids = pd.read_csv(path, nrows=0).columns[1:]

n = len(gpu_ids)

# One counter for each GPU
valid_count = np.zeros(n, dtype=int)
missing_count = np.zeros(n, dtype=int)
invalid_count = np.zeros(n, dtype=int)
power_sum = np.zeros(n)

# Read 1000 timestamps at a time
for chunk in pd.read_csv(
    path,
    chunksize=1000,
    dtype=str,
    keep_default_na=False
):

    # Remove timestamp column
    raw = chunk.iloc[:, 1:]

    # Empty fields represent missing measurements
    missing = raw.eq("")

    # Convert measurements into numbers
    values = raw.apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    # Valid power is finite and non-negative
    valid = np.isfinite(values) & (values >= 0)

    # Update the statistics for each GPU
    valid_count += valid.sum(axis=0)
    missing_count += missing.sum(axis=0).to_numpy()
    invalid_count += len(chunk) - valid.sum(axis=0) - missing.sum(axis=0).to_numpy()

    power_sum += np.where(valid, values, 0).sum(axis=0)


# Calculate average power
avg_power = np.divide(
    power_sum,
    valid_count,
    out=np.full(n, np.nan),
    where=valid_count > 0
)

# Save the final GPU summary
power_result = pd.DataFrame({
    "gpu_id": gpu_ids,
    "valid_count": valid_count,
    "missing_count": missing_count,
    "invalid_count": invalid_count,
    "avg_power_w": np.round(avg_power, 2)
})

power_result.to_parquet(
    "benchmarks/seren_power_pandas.parquet",
    index=False
)

power_time = time.perf_counter() - start

print("Power analysis completed.")
print("GPUs analyzed:", len(power_result))


# ==========================================
# PART 2 — JOB ANALYSIS
# ==========================================

start = time.perf_counter()

job_path = "data/raw/data/job_trace/trace_seren.csv"

# We only need the number of requested GPUs
jobs = pd.read_csv(job_path, usecols=["gpu_num"])

# Convert GPU requests into numbers
gpu_requests = pd.to_numeric(
    jobs["gpu_num"],
    errors="coerce"
)

# Identify valid GPU jobs
gpu_jobs = (gpu_requests > 0) & np.isfinite(gpu_requests)

# Calculate simple workload statistics
job_result = pd.DataFrame([{
    "total_jobs": len(jobs),
    "gpu_jobs": int(gpu_jobs.sum()),
    "requested_gpu_sum": float(gpu_requests[gpu_jobs].sum())
}])

job_result.to_parquet(
    "benchmarks/seren_jobs_pandas.parquet",
    index=False
)

job_time = time.perf_counter() - start


# ==========================================
# PART 3 — DISPLAY EXECUTION TIMES
# ==========================================

print("\n=== PANDAS RESULTS ===")
print(job_result.to_string(index=False))

print(f"\nPower processing: {power_time:.2f} seconds")
print(f"Job processing:   {job_time:.2f} seconds")
print(f"Total:            {power_time + job_time:.2f} seconds")
