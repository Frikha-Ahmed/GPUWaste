# GPUWaste
### Distributed Detection of GPU Underutilization and Resource Inefficiency in Large AI Clusters

GPUWaste is an academic Big Data project that analyzes GPU utilization,
power measurements, and job execution traces from real AI clusters.

The system uses Apache Hadoop HDFS, Apache Spark, MongoDB, and Streamlit
to process large-scale monitoring data and explore potential resource
inefficiency.

## 1. Problem Statement

Modern AI clusters contain thousands of GPUs running training,
evaluation, and other computational workloads.

However, GPU resources may experience prolonged periods of low
utilization while still consuming electrical power.

Analyzing these patterns is challenging because monitoring systems
generate large volumes of time-series data across many GPUs.

GPUWaste addresses this problem using distributed data processing.

The objective is to:
- Analyze GPU utilization across large AI clusters.
- Identify potential underutilization patterns.
- Explore GPU power consumption.
- Analyze job execution statistics.
- Compare the behavior of different GPU clusters.
- Visualize analytical results through an interactive dashboard.

GPUWaste is a historical batch-analytics system, not a real-time
monitoring or GPU scheduling platform.

## 2. Dataset

GPUWaste uses AcmeTrace, a dataset containing real GPU cluster workloads.

Dataset:
https://huggingface.co/datasets/HuggingAGree/AcmeTrace

Project:
https://github.com/InternLM/AcmeTrace

We analyze two clusters:

| Cluster | Monitored GPUs | Main purpose |
|---------|---------------:|--------------|
| Kalos | 2,344 | GPU analytics, underutilization and temporal analysis |
| Seren | 1,992 | GPU analytics, workload analysis and scalability evaluation |

The monitoring data includes:
- GPU utilization measurements.
- GPU power measurements.
- Timestamps.
- Job execution traces.

The original datasets are not included in this repository.

## 3. Big Data Architecture

The system follows this architecture:

    AcmeTrace Dataset
      |
      +------------------+
      |                  |
      v                  v
    Kalos              Seren
      |                  |
      +--------+---------+
               |
               v
       Apache Hadoop HDFS
       (1 NameNode, 3 DataNodes)
               |
               v
          Apache Spark
       (1 Master, 2 Workers)
               |
               v
       Distributed Analytics
               |
               v
         Parquet in HDFS
               |
               v
            MongoDB
               |
               v
           Streamlit
               |
      +--------+---------+
      |        |         |
      v        v         v
    Kalos    Seren    Comparison
    Dashboard Dashboard Dashboard

### Why these technologies?

**Hadoop HDFS**
- Stores large monitoring datasets.
- Distributes files across DataNodes.
- Supports block replication.

**Apache Spark**
- Processes large datasets in parallel.
- Supports RDD and DataFrame operations.
- Performs distributed aggregations.

**Apache Parquet**
- Stores structured analytical results.
- Provides efficient column-oriented storage.

**MongoDB**
- Stores analytical summaries as documents.
- Allows the dashboard to query results without rerunning Spark.

**Streamlit**
- Provides an interactive Python-based dashboard.
- Displays metrics, charts, and analytical tables.

**Docker Compose**
- Deploys the Hadoop, Spark, and MongoDB services.

The distributed services were deployed as containers on one physical
computer. Therefore, HDFS replication was tested across containers,
not across independent physical machines.

## 4. Main Features

### Kalos Dashboard

- GPU utilization statistics.
- Hourly utilization trends.
- Underutilization candidate detection.
- GPU power consumption statistics.
- Timestamp-aligned power and utilization analysis for candidates.
- Job execution statistics.

### Seren Dashboard

- GPU utilization statistics.
- Hourly utilization trends.
- GPU explorer.
- GPU power consumption analysis.
- Job execution statistics.
- Underutilization candidate detection.

### Cluster Comparison

- Number of monitored GPUs.
- Average GPU utilization.
- Underutilization candidate counts.
- Comparison charts.

## 5. Underutilization Detection

GPUWaste applies the following criteria to identify potential
underutilization candidates:

| Criterion | Threshold |
|-----------|-----------|
| Average GPU utilization | < 10% |
| Percentage of low-utilization observations | >= 80% |
| Monitoring coverage | >= 80% |
| Valid monitoring observations | >= 1,000 |

The same candidate-selection criteria are applied to Kalos and Seren.

Missing measurements are not treated as zero utilization.

The thresholds are analytical heuristics rather than validated
operational definitions of energy waste.

## 6. Experimental Results

### GPU Monitoring

| Metric | Kalos | Seren |
|--------|------:|------:|
| Monitored GPUs | 2,344 | 1,992 |
| Underutilization candidates | 196 | 0 |

Seren has no GPUs satisfying all four detection criteria.

This does not imply that Seren has no resource inefficiency.

### Seren Job Statistics

| Metric | Value |
|--------|------:|
| Total jobs | 818,327 |
| Jobs requesting GPUs | 450,590 |
| Sum of GPU requests across jobs | 2,917,257 |
| Hourly monitoring records | 2,246 |

The sum of GPU requests is calculated across jobs. It does not
represent the number of distinct physical GPUs.

## 7. Performance Benchmarks

We compared local Pandas implementations with Spark implementations
running in our containerized distributed environment.

| Workload | Pandas | Spark (2 executors) |
|----------|-------:|--------------------:|
| Kalos GPU utilization (~843 MB) | 5.318 s | 30.104 s |
| Seren GPU utilization (~5.1 GB) | 236.852 s | 68.562 s |
| Seren power + job analysis | 276.192 s | 67.254 s |

For the larger Seren workloads, Spark was faster in these experiments.

For the smaller Kalos utilization workload, Pandas was faster.

These results describe our measured implementations, not a general
performance guarantee. The Pandas and Spark implementations use
different execution and storage paths, and all benchmarks ran
on one physical computer.

## 8. Project Structure

    GPUWaste/
    |
    +-- docker/
    |   +-- compose.yml
    |   +-- compose.spark.yml
    |   +-- compose.mongo.yml
    |   +-- Dockerfile.hadoop
    |   +-- Dockerfile.spark-client
    |
    +-- src/
    |   +-- 01_jobs_etl.py
    |   +-- 02_gpu_utilization.py
    |   +-- 03_gpu_hourly.py
    |   +-- 04_gpu_inefficiency.py
    |   +-- 05_gpu_power.py
    |   +-- 06_gpu_efficiency.py
    |   +-- 07_gpu_temporal_analysis.py
    |   +-- 08_export_mongodb.py
    |   +-- 09_dashboard.py
    |   +-- 10_prepare_seren.py
    |   +-- 10_seren_hourly.py
    |   +-- 11_export_seren.py
    |   +-- 12_dashboard.py
    |
    +-- benchmarks/
    |
    +-- README.md

The `src/09_dashboard.py` file contains the original Kalos dashboard.

The new `src/12_dashboard.py` is the main application entry point
and provides navigation between Kalos, Seren, and cluster comparison.

## 9. Running GPUWaste

### Requirements

- Linux with Docker and Docker Compose.
- Python 3 and pip.
- Sufficient disk space for the selected AcmeTrace files.

The project uses containerized Hadoop and Spark services.
A local GPU is not required to analyze historical trace files.

### Clone the repository

```bash
git clone https://github.com/Frikha-Ahmed/GPUWaste.git
cd GPUWaste
```

### Configure MongoDB

The file `docker/mongodb.env` contains local MongoDB credentials
and must not be committed to Git.

It must define:

    MONGO_INITDB_ROOT_USERNAME
    MONGO_INITDB_ROOT_PASSWORD

For an existing installation, retain the original credentials
associated with the persistent MongoDB volume.

### Start the infrastructure

```bash
sudo docker compose \
  -f docker/compose.yml \
  -f docker/compose.spark.yml \
  -f docker/compose.mongo.yml \
  up -d
```

Check the containers:

```bash
sudo docker compose \
  -f docker/compose.yml \
  -f docker/compose.spark.yml \
  -f docker/compose.mongo.yml \
  ps
```

### Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install streamlit pandas pymongo python-dotenv pyarrow
```

### Launch the dashboard

Once the analytical results have been generated and exported to
MongoDB, run:

```bash
python -m streamlit run src/12_dashboard.py
```

Open:

http://localhost:8501

### Web interfaces

| Component | Address |
|-----------|---------|
| Hadoop NameNode | http://localhost:9870 |
| Spark Master | http://localhost:8080 |
| Streamlit | http://localhost:8501 |

Important: Starting the containers alone does not recreate the
analytical database. Raw datasets must first be loaded into HDFS,
processed by Spark, and exported to MongoDB.

## 10. Data Processing Workflow

The original Kalos pipeline includes:

1. Job data transformation.
2. GPU utilization aggregation.
3. Hourly utilization aggregation.
4. Underutilization candidate detection.
5. GPU power analysis.
6. GPU utilization and power integration.
7. Timestamp-aligned candidate analysis.
8. MongoDB export.

For Seren, the implementation reuses previously generated Spark
benchmark summaries for GPU utilization and power.

The additional Seren modules are:

- `10_prepare_seren.py`: prepares per-GPU summaries, candidate
  detection results, and job statistics.
- `10_seren_hourly.py`: calculates hourly utilization statistics.
- `11_export_seren.py`: exports Seren summaries to MongoDB.
- `12_dashboard.py`: provides the dual-cluster interface.

`10_prepare_seren.py` requires the previously generated Seren
benchmark Parquet outputs in HDFS.

`11_export_seren.py` reads the Seren analytical Parquet directories
from `data/seren_export/` on the host computer.

A fresh installation must reproduce these intermediate results
before the dual-cluster dashboard can display data.

## 11. Scientific Limitations

- Low GPU utilization does not automatically establish energy waste.
- GPU power measurements and utilization must be aligned by timestamp
  before interpreting their relationship at a specific moment.
- Historical averages do not establish simultaneous activity.
- Detection thresholds are configurable analytical assumptions.
- The project does not automatically reschedule workloads.
- The deployment demonstrates distributed processing on one host,
  not physical multi-machine fault tolerance.
- Benchmark results depend on the workloads and execution environment.

## 12. Academic Context

GPUWaste is an academic Big Data engineering project developed
at ENSI, Tunisia.

Its purpose is to demonstrate a complete data-processing pipeline,
from large-scale storage and distributed analytics to NoSQL
persistence and interactive visualization.
