import csv
import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. Start Spark
spark = SparkSession.builder.appName("GPUWaste-Hourly").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# 2. Read our GPU data from HDFS
path = "hdfs://namenode:8020/gpuwaste/raw/kalos/GPU_UTIL.csv"

lines = spark.sparkContext.textFile(path)

# Remove the CSV header
data = lines.filter(lambda line: not line.startswith("Time,"))


# 3. Analyze one timestamp (one CSV row)
def analyze_row(line):

    row = next(csv.reader([line]))

    # Group measurements by hour, in the source timezone (+08:00)
    hour = row[0][:13]

    total = 0.0
    valid = 0
    low = 0
    missing = 0
    invalid = 0

    # Examine every GPU measurement in the row
    for value in row[1:]:

        if value.strip() == "":
            missing += 1
            continue

        try:
            usage = float(value)
        except ValueError:
            invalid += 1
            continue

        if not math.isfinite(usage) or not 0 <= usage <= 100:
            invalid += 1
            continue

        total += usage
        valid += 1

        if usage < 10:
            low += 1

    return (hour, (total, valid, low, missing, invalid))


# 4. Combine statistics belonging to the same hour
def combine(a, b):
    return (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4]
    )


hourly = data.map(analyze_row).reduceByKey(combine)


# 5. Convert the results into a Spark DataFrame
schema = """
    hour STRING,
    utilization_sum DOUBLE,
    valid_count LONG,
    low_count LONG,
    missing_count LONG,
    invalid_count LONG
"""

df = spark.createDataFrame(
    hourly.map(lambda x: (x[0],) + x[1]),
    schema=schema
)


# 6. Calculate hourly indicators
result = (
    df.withColumn(
        "avg_utilization",
        F.when(
            F.col("valid_count") > 0,
            F.round(F.col("utilization_sum") / F.col("valid_count"), 2)
        )
    )
    .withColumn(
        "low_utilization_pct",
        F.when(
            F.col("valid_count") > 0,
            F.round(100 * F.col("low_count") / F.col("valid_count"), 2)
        )
    )
    .drop("utilization_sum")
)

# Keep the small aggregated result for reuse
result.cache()


# 7. Save hourly statistics to HDFS
output = "hdfs://namenode:8020/gpuwaste/gold/gpu_hourly"

result.write.mode("overwrite").parquet(output)


# 8. Display the results
print("\n=== HOURLY GPU UTILIZATION ===")

result.orderBy("hour").show(20, truncate=False)

spark.stop()
