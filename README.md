# GPUWaste

### Distributed Detection of GPU Underutilization in Large AI Clusters

GPUWaste is a Big Data project developed to analyze GPU resource
utilization and potential resource inefficiency in large AI clusters.

## Problem

AI clusters contain thousands of GPUs running different workloads.
Some GPUs exhibit prolonged periods of low computational activity
while continuing to consume electrical power.

GPUWaste investigates these patterns using distributed analytics.

## Dataset

AcmeTrace — Real GPU Cluster Workloads

Source:
https://huggingface.co/datasets/HuggingAGree/AcmeTrace

Two production clusters:
- Kalos
- Seren

## Architecture

AcmeTrace
    |
    v
Hadoop HDFS (3 DataNodes, replication factor 3)
    |
    v
Apache Spark (2 Workers)
    |
    v
GPU Utilization and Power Analytics
    |
    v
Apache Parquet
    |
    v
MongoDB
    |
    v
Streamlit Dashboard

## Technologies

- Python
- Apache Hadoop HDFS
- Apache Spark / PySpark
- Apache Parquet
- MongoDB
- Streamlit
- Docker Compose
- Pandas and NumPy

## Main Features

- Distributed GPU utilization analysis
- GPU power consumption monitoring
- Potential GPU underutilization detection
- Timestamp-aligned GPU power analysis
- AI workload statistics
- Interactive Streamlit dashboard
- Python/Pandas versus Apache Spark benchmarks

## Project Structure

- src/ : Python processing modules and dashboard
- docker/ : Docker and distributed infrastructure configuration
- benchmarks/ : Performance evaluation results

## Important Note

The AcmeTrace datasets are not included in this repository.
They must be downloaded separately from Hugging Face.

GPUWaste identifies potential underutilization patterns.
Its indicators do not establish confirmed energy waste or
recoverable energy savings.

## Academic Project

Big Data — Engineering Project
ENSI, Tunisia
