from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ==========================================
# STEP 1 — Start Spark
# ==========================================

spark = SparkSession.builder.appName("GPUWaste-Efficiency").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# ==========================================
# STEP 2 — Read our previous results
# ==========================================

# GPUs detected with consistently low utilization
gpu_path = "hdfs://namenode:8020/gpuwaste/gold/gpu_low_utilization_candidates"

gpu = spark.read.parquet(gpu_path)


# Power consumption statistics
power_path = "hdfs://namenode:8020/gpuwaste/gold/gpu_power_summary"

power = spark.read.parquet(power_path)


# ==========================================
# STEP 3 — Select the power columns we need
# ==========================================

power = power.select(
    "gpu_id",
    "avg_power_w",
    "max_power_w",
    F.col("coverage_pct").alias("power_coverage_pct")
)


# ==========================================
# STEP 4 — Combine both datasets
# ==========================================

# Match records using the same GPU identifier
combined = gpu.join(power, on="gpu_id", how="inner")


# ==========================================
# STEP 5 — Keep reliable power measurements
# ==========================================

# Require at least 80% valid power measurements
result = combined.filter(
    (F.col("power_coverage_pct") >= 80) &
    F.col("avg_power_w").isNotNull()
)


# ==========================================
# STEP 6 — Select final indicators
# ==========================================

result = result.select(
    "gpu_id",
    "avg_utilization",
    "low_utilization_pct",
    "coverage_pct",
    "avg_power_w",
    "max_power_w",
    "power_coverage_pct"
)


# ==========================================
# STEP 7 — Save results in HDFS
# ==========================================

output = "hdfs://namenode:8020/gpuwaste/gold/gpu_efficiency"

result.write.mode("overwrite").parquet(output)


# ==========================================
# STEP 8 — Display results
# ==========================================

print("\n=== GPUWASTE RESOURCE EFFICIENCY ===")

print("Initially detected GPUs:", gpu.count())
print("GPUs with sufficient power data:", result.count())

# Display candidates with the highest average power
result.orderBy(
    F.desc("avg_power_w")
).show(20, truncate=False)


# ==========================================
# STEP 9 — Stop Spark
# ==========================================

spark.stop()
