from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# -------------------------------------------------
# 1. Create Spark session
# -------------------------------------------------

spark = (
    SparkSession.builder
    .appName("GPUWaste-JobTrace-ETL")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)


# -------------------------------------------------
# 2. Read job traces from Hadoop HDFS
# -------------------------------------------------

input_path = (
    "hdfs://namenode:8020/"
    "gpuwaste/raw/kalos/trace_kalos.csv"
)

raw = (
    spark.read
    .option("header", True)
    .option("mode", "PERMISSIVE")
    .csv(input_path)
)


# -------------------------------------------------
# 3. Clean and transform data
# -------------------------------------------------

jobs = raw.select(

    F.col("job_id"),

    F.col("user").alias("user_id"),

    F.col("node_num").cast("int").alias("node_num"),

    F.col("gpu_num").cast("int").alias("gpu_num"),

    F.col("cpu_num").cast("int").alias("cpu_num"),

    F.col("type"),

    F.col("state"),

    F.to_timestamp(
        "submit_time",
        "yyyy-MM-dd HH:mm:ssXXX"
    ).alias("submit_ts"),

    F.to_timestamp(
        "start_time",
        "yyyy-MM-dd HH:mm:ssXXX"
    ).alias("start_ts"),

    F.to_timestamp(
        "end_time",
        "yyyy-MM-dd HH:mm:ssXXX"
    ).alias("end_ts"),

    F.col("duration").cast("double").alias("duration_s"),

    F.col("queue").cast("double").alias("queue_s"),

    F.col("gpu_time").cast("double").alias("gpu_time")

)

jobs = jobs.filter(
    F.col("job_id").isNotNull()
)

jobs = jobs.withColumn(
    "event_month",
    F.date_format(
        F.coalesce("start_ts", "submit_ts"),
        "yyyy-MM"
    )
)


# -------------------------------------------------
# 4. Calculate workload statistics
# -------------------------------------------------

summary = (
    jobs.groupBy("type", "state")
    .agg(
        F.count("*").alias("total_jobs"),

        F.avg("gpu_num").alias("avg_requested_gpus"),

        F.avg("duration_s").alias("avg_duration_seconds"),

        F.avg("queue_s").alias("avg_queue_seconds"),

        F.sum("gpu_time").alias("total_reported_gpu_time")
    )
)


# -------------------------------------------------
# 5. Save cleaned data to HDFS (Silver layer)
# -------------------------------------------------

jobs.write \
    .mode("overwrite") \
    .partitionBy("event_month") \
    .parquet(
        "hdfs://namenode:8020/gpuwaste/silver/jobs"
    )


# -------------------------------------------------
# 6. Save analytical results (Gold layer)
# -------------------------------------------------

summary.write \
    .mode("overwrite") \
    .parquet(
        "hdfs://namenode:8020/"
        "gpuwaste/gold/job_summary"
    )


# -------------------------------------------------
# 7. Display processing results
# -------------------------------------------------

print("=== GPUWASTE JOB ANALYSIS ===")

summary.orderBy(
    F.desc("total_jobs")
).show(30, truncate=False)

spark.stop()
