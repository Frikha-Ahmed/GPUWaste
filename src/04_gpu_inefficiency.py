from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================
# STEP 1: Start Spark
# ============================================

spark = SparkSession.builder.appName("GPUWaste-Detection").getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================
# STEP 2: Read our previous GPU analysis
# ============================================

input_path = "hdfs://namenode:8020/gpuwaste/gold/gpu_utilization_summary"

df = spark.read.parquet(input_path)


# ============================================
# STEP 3: Calculate data coverage
# ============================================

# Total number of expected measurements
df = df.withColumn(
    "total_samples",
    F.col("valid_count") +
    F.col("missing_count") +
    F.col("invalid_count")
)

# Percentage of available and valid measurements
df = df.withColumn(
    "coverage_pct",
    F.round(100 * F.col("valid_count") / F.col("total_samples"), 2)
)


# ============================================
# STEP 4: Detect potentially underutilized GPUs
# ============================================

# A GPU is selected when it satisfies ALL conditions
candidates = df.filter(
    (F.col("avg_utilization") < 10) &
    (F.col("low_utilization_pct") >= 80) &
    (F.col("coverage_pct") >= 80) &
    (F.col("valid_count") >= 1000)
)


# ============================================
# STEP 5: Select useful columns
# ============================================

candidates = candidates.select(
    "gpu_id",
    "avg_utilization",
    "low_utilization_pct",
    "coverage_pct",
    "valid_count",
    "missing_count"
)


# ============================================
# STEP 6: Save detection results to HDFS
# ============================================

output_path = "hdfs://namenode:8020/gpuwaste/gold/gpu_low_utilization_candidates"

candidates.write.mode("overwrite").parquet(output_path)


# ============================================
# STEP 7: Display our results
# ============================================

print("\n=== GPUWASTE DETECTION RESULTS ===")

print("Total monitored GPUs:", df.count())

print("Potentially underutilized GPUs:", candidates.count())

candidates.orderBy("avg_utilization").show(20, truncate=False)


# ============================================
# STEP 8: Stop Spark
# ============================================

spark.stop()
