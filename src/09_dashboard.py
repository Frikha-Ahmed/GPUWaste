import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="GPUWaste",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GPUWaste")
st.write("Distributed GPU Resource Efficiency Monitoring")
st.caption("AcmeTrace — Shanghai AI Laboratory | Kalos Cluster")


# ==================================================
# 2. CONNECT TO MONGODB
# ==================================================

# Load the credentials from our local configuration file
load_dotenv("docker/mongodb.env")


@st.cache_resource
def connect_mongodb():

    # Connect to the MongoDB container from Ubuntu
    client = MongoClient(
        host="localhost",
        port=27017,
        username=os.getenv("MONGO_INITDB_ROOT_USERNAME"),
        password=os.getenv("MONGO_INITDB_ROOT_PASSWORD"),
        authSource="admin",
        serverSelectionTimeoutMS=3000
    )

    # Verify that MongoDB is available
    client.admin.command("ping")

    return client["gpuwaste"]


try:
    db = connect_mongodb()

except Exception as error:
    st.error(f"Cannot connect to MongoDB: {error}")
    st.stop()


# ==================================================
# 3. LOAD ANALYTICAL RESULTS
# ==================================================

@st.cache_data(ttl=60)
def load_data(collection_name):

    # Read the documents without MongoDB's internal _id
    documents = list(
        db[collection_name].find({}, {"_id": 0})
    )

    # Convert documents into a Pandas DataFrame
    return pd.DataFrame(documents)


gpu_data = load_data("gpu_utilization")
candidates = load_data("gpu_efficiency")
hourly = load_data("gpu_hourly")
jobs = load_data("job_summary")
temporal = load_data("gpu_temporal")


# Refresh results when new data is exported
if st.button("Refresh data"):
    load_data.clear()
    st.rerun()


# ==================================================
# 4. DISPLAY KEY INDICATORS
# ==================================================

total_gpus = len(gpu_data)
detected_gpus = len(candidates)

candidate_pct = (
    100 * detected_gpus / total_gpus
    if total_gpus > 0 else 0
)

col1, col2, col3 = st.columns(3)

col1.metric("Monitored GPUs", total_gpus)
col2.metric("Underutilization Candidates", detected_gpus)
col3.metric("Candidate Percentage", f"{candidate_pct:.2f}%")


# ==================================================
# 5. CREATE DASHBOARD TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "GPU Activity",
    "GPU Efficiency",
    "Workload Analysis"
])


# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with tab1:

    st.subheader("GPUWaste Overview")

    st.write(
        "This dashboard analyzes GPU utilization, "
        "power measurements, and AI workloads "
        "using results processed by Apache Spark."
    )

    st.subheader("Top Candidates by Average Power")

    if not candidates.empty:

        top_gpus = candidates.sort_values(
            "avg_power_w",
            ascending=False
        ).head(10)

        st.dataframe(top_gpus, hide_index=True)

    st.info(
        "Low GPU utilization indicates potential "
        "resource inefficiency, not confirmed waste."
    )


# ==================================================
# TAB 2 — GPU ACTIVITY OVER TIME
# ==================================================

with tab2:

    st.subheader("Hourly GPU Utilization")

    if not hourly.empty:

        # Convert hour strings into datetime values
        hourly["time"] = pd.to_datetime(
            hourly["hour"],
            format="%Y-%m-%d %H"
        )

        # Sort observations chronologically
        hourly = hourly.sort_values("time")

        # Display average GPU utilization over time
        st.line_chart(
            hourly.set_index("time")["avg_utilization"]
        )

        st.caption(
            "Timestamps are displayed in the "
            "original dataset timezone (UTC+08:00)."
        )

        st.subheader("Hourly Statistics")

        st.dataframe(hourly, hide_index=True)


# ==================================================
# TAB 3 — GPU EFFICIENCY
# ==================================================

with tab3:

    st.subheader("Potentially Underutilized GPUs")

    if not candidates.empty:

        st.dataframe(candidates, hide_index=True)

        # Select an individual GPU
        selected_gpu = st.selectbox(
            "Select a GPU to investigate",
            candidates["gpu_id"].tolist()
        )

        # Retrieve the selected GPU's statistics
        gpu_info = candidates[
            candidates["gpu_id"] == selected_gpu
        ].iloc[0]

        st.subheader("Selected GPU")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Utilization",
            f"{gpu_info['avg_utilization']:.2f}%"
        )

        col2.metric(
            "Average Power",
            f"{gpu_info['avg_power_w']:.2f} W"
        )

        col3.metric(
            "Monitoring Coverage",
            f"{gpu_info['coverage_pct']:.2f}%"
        )

        # Display timestamp-aligned power analysis
        gpu_temporal = temporal[
            temporal["gpu_id"] == selected_gpu
        ]

        if not gpu_temporal.empty:

            st.subheader("Temporal Power Analysis")

            st.dataframe(gpu_temporal, hide_index=True)

    st.warning(
        "These indicators do not establish "
        "actual GPU allocation waste or "
        "recoverable energy savings."
    )


# ==================================================
# TAB 4 — AI WORKLOAD ANALYSIS
# ==================================================

with tab4:

    st.subheader("AI Workload Statistics")

    if not jobs.empty:

        # Count jobs by workload category
        job_counts = (
            jobs.groupby("type")["total_jobs"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(job_counts)

        st.subheader("Detailed Job Statistics")

        st.dataframe(
            jobs.sort_values(
                "total_jobs",
                ascending=False
            ),
            hide_index=True
        )


# ==================================================
# 6. FOOTER
# ==================================================

st.divider()

st.caption(
    "GPUWaste | Hadoop HDFS + Apache Spark + "
    "MongoDB + Streamlit"
)
