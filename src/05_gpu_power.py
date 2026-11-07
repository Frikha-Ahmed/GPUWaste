from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# STEP 1 — Start Spark
spark = SparkSession.builder.appName("GPUWaste-Power").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# STEP 2 — Read the power dataset from HDFS
path = "hdfs://namenode:8020/gpuwaste/raw/kalos/POWER_USAGE.csv"

df = spark.read.csv(path, header=True)


# STEP 3 — Identify the GPU columns
# The first column is Time. All the others are GPU identifiers.
gpu_columns = []

for gpu_name in df.columns[1:]:
    # Backticks allow Spark to recognize names containing dots.
    gpu_columns.append(F.col(f"`{gpu_name}`"))


# STEP 4 — Convert GPU columns into rows
# Before: Time | GPU_A | GPU_B
# After:  Time | gpu_id | raw_power
df = df.unpivot("Time", gpu_columns, "gpu_id", "raw_power")


# STEP 5 — Convert power measurements into numbers
df = df.withColumn(
    "power_w",
    F.col("raw_power").cast("double")
)


# STEP 6 — Define valid and missing measurements

# A valid measurement must be a finite, non-negative number.
valid = (
    F.col("power_w").isNotNull()
    & (F.col("power_w") >= 0)
    & (~F.isnan("power_w"))
    & (F.col("power_w") < float("inf"))
)

# Missing means the original CSV field is empty.
missing = (
    F.col("raw_power").isNull()
    | (F.trim(F.col("raw_power")) == "")
)

# Keep valid power values; replace other values with NULL.
valid_power = F.when(valid, F.col("power_w"))


# STEP 7 — Calculate statistics for every GPU
result = df.groupBy("gpu_id").agg(
    F.count("*").alias("total_samples"),
    F.count(valid_power).alias("valid_count"),
    F.count(F.when(missing, 1)).alias("missing_count"),
    F.round(F.avg(valid_power), 2).alias("avg_power_w"),
    F.round(F.max(valid_power), 2).alias("max_power_w")
)


# STEP 8 — Calculate additional indicators
result = result.withColumn(
    "invalid_count",
    F.col("total_samples") - F.col("valid_count") - F.col("missing_count")
)

result = result.withColumn(
    "coverage_pct",
    F.round(100 * F.col("valid_count") / F.col("total_samples"), 2)
)


# STEP 9 — Save results in HDFS
result.cache()

output = "hdfs://namenode:8020/gpuwaste/gold/gpu_power_summary"

result.write.mode("overwrite").parquet(output)


# STEP 10 — Display results
print("\n=== GPU POWER ANALYSIS ===")

result.orderBy(F.desc("avg_power_w")).show(20, truncate=False)

spark.stop()
