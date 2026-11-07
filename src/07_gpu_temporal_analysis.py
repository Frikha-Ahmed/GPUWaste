from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# STEP 1: Start Spark
spark = SparkSession.builder.appName("GPUWaste-Temporal").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# STEP 2: Get the 196 GPUs detected previously
candidates_path = "hdfs://namenode:8020/gpuwaste/gold/gpu_efficiency"

candidates = spark.read.parquet(candidates_path)

gpu_ids = [row["gpu_id"] for row in candidates.select("gpu_id").collect()]

print("GPUs to analyze:", len(gpu_ids))


# STEP 3: Read GPU utilization
util_path = "hdfs://namenode:8020/gpuwaste/raw/kalos/GPU_UTIL.csv"

util = spark.read.csv(util_path, header=True)

# Select only the GPU columns we need.
# Backticks are necessary because GPU names contain dots.
gpu_columns = [F.col(f"`{gpu_id}`") for gpu_id in gpu_ids]

# Convert GPU columns into rows
util = util.unpivot("Time", gpu_columns, "gpu_id", "raw_util")

# Convert utilization into numbers
util = util.withColumn(
    "utilization",
    F.col("raw_util").cast("double")
)

# Keep valid utilization measurements
util = util.filter(
    F.col("utilization").between(0, 100)
    & (~F.isnan("utilization"))
)

# Keep only the columns needed for the join
util = util.select("Time", "gpu_id", "utilization")


# STEP 4: Read GPU power measurements
power_path = "hdfs://namenode:8020/gpuwaste/raw/kalos/POWER_USAGE.csv"

power = spark.read.csv(power_path, header=True)

# Convert GPU columns into rows
power = power.unpivot("Time", gpu_columns, "gpu_id", "raw_power")

# Convert power values into numbers
power = power.withColumn(
    "power_w",
    F.col("raw_power").cast("double")
)

# Keep valid power measurements
power = power.filter(
    (F.col("power_w") >= 0)
    & (F.col("power_w") < float("inf"))
    & (~F.isnan("power_w"))
)

power = power.select("Time", "gpu_id", "power_w")


# STEP 5: Match utilization and power measurements
# Both GPU ID and timestamp must match.
combined = util.join(
    power,
    on=["Time", "gpu_id"],
    how="inner"
)


# STEP 6: Calculate power consumption during low utilization

# A measurement is considered low utilization below 10%
low = F.col("utilization") < 10

result = combined.groupBy("gpu_id").agg(

    # Number of matched measurements
    F.count("*").alias("matched_samples"),

    # Number of measurements with low utilization
    F.count(F.when(low, 1)).alias("low_samples"),

    # Average power when utilization is below 10%
    F.round(
        F.avg(F.when(low, F.col("power_w"))), 2
    ).alias("power_when_low_w"),

    # Average power when utilization is at least 10%
    F.round(
        F.avg(F.when(~low, F.col("power_w"))), 2
    ).alias("power_when_other_w")
)


# STEP 7: Calculate percentage of low-utilization observations
result = result.withColumn(
    "low_pct",
    F.round(
        100 * F.col("low_samples") / F.col("matched_samples"), 2
    )
)


# STEP 8: Save results in HDFS
output = "hdfs://namenode:8020/gpuwaste/gold/gpu_temporal_analysis"

result.write.mode("overwrite").parquet(output)


# STEP 9: Display saved results
print("\n=== GPU TEMPORAL ANALYSIS ===")

saved = spark.read.parquet(output)

saved.orderBy(F.desc("power_when_low_w")).show(20, truncate=False)

spark.stop()
