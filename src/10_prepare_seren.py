from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ==========================================
# 1. START SPARK
# ==========================================

spark = SparkSession.builder.appName("GPUWaste-Seren").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

base = "hdfs://namenode:8020/gpuwaste"


# ==========================================
# 2. READ EXISTING SEREN BENCHMARK RESULTS
# ==========================================

# Reuse our previous GPU utilization analysis
util = spark.read.parquet(
    f"{base}/gold/benchmark_seren_spark"
)

# Reuse our previous power consumption analysis
power = spark.read.parquet(
    f"{base}/gold/benchmark_seren_power"
)


# ==========================================
# 3. CALCULATE GPU MONITORING COVERAGE
# ==========================================

util = util.withColumn(
    "total_samples",
    F.col("valid_count") +
    F.col("missing_count") +
    F.col("invalid_count")
)

util = util.withColumn(
    "coverage_pct",
    F.round(
        100 * F.col("valid_count") / F.col("total_samples"),
        2
    )
)


# ==========================================
# 4. DETECT UNDERUTILIZED GPU CANDIDATES
# ==========================================

# Apply the same four rules as Kalos
candidates = util.filter(
    (F.col("avg_utilization") < 10) &
    (F.col("low_utilization_pct") >= 80) &
    (F.col("coverage_pct") >= 80) &
    (F.col("valid_count") >= 1000)
)


# ==========================================
# 5. PREPARE POWER STATISTICS
# ==========================================

power = power.withColumn(
    "total_samples",
    F.col("valid_count") +
    F.col("missing_count") +
    F.col("invalid_count")
)

power = power.withColumn(
    "coverage_pct",
    F.round(
        100 * F.col("valid_count") / F.col("total_samples"),
        2
    )
)


# ==========================================
# 6. COMBINE UTILIZATION AND POWER
# ==========================================

# Rename power coverage to avoid duplicate names
power_selected = power.select(
    "gpu_id",
    "avg_power_w",
    F.col("coverage_pct").alias("power_coverage_pct")
)

# Match candidates with their corresponding power data
efficiency = candidates.join(
    power_selected,
    on="gpu_id",
    how="inner"
)

efficiency = efficiency.filter(
    F.col("power_coverage_pct") >= 80
)


# ==========================================
# 7. ANALYZE SEREN JOB HISTORY
# ==========================================

jobs = spark.read.csv(
    f"{base}/raw/seren/trace_seren.csv",
    header=True
)

# Convert requested GPUs into a number
jobs = jobs.withColumn(
    "gpu_num",
    F.col("gpu_num").cast("double")
)

# Calculate basic job statistics
job_summary = jobs.agg(
    F.count("*").alias("total_jobs"),

    F.sum(
        F.when(F.col("gpu_num") > 0, 1).otherwise(0)
    ).alias("gpu_jobs"),

    F.sum(
        F.when(F.col("gpu_num") > 0, F.col("gpu_num"))
        .otherwise(0)
    ).alias("requested_gpu_sum")
)


# ==========================================
# 8. SAVE RESULTS TO HDFS
# ==========================================

util.write.mode("overwrite").parquet(
    f"{base}/gold/seren_gpu_utilization"
)

power.write.mode("overwrite").parquet(
    f"{base}/gold/seren_gpu_power"
)

candidates.write.mode("overwrite").parquet(
    f"{base}/gold/seren_gpu_candidates"
)

efficiency.write.mode("overwrite").parquet(
    f"{base}/gold/seren_gpu_efficiency"
)

job_summary.write.mode("overwrite").parquet(
    f"{base}/gold/seren_job_summary"
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n=== SEREN ANALYTICS ===")

print("Monitored GPUs:", util.count())
print("Underutilization candidates:", candidates.count())

job_summary.show(truncate=False)

spark.stop()
