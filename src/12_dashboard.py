"""GPUWaste multi-cluster dashboard entrypoint.

Preserves src/09_dashboard.py as the Kalos page and adds Seren and comparison.
Run from ~/GPUWaste: python -m streamlit run src/12_dashboard.py
"""
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from pymongo import MongoClient

PROJECT = Path(__file__).resolve().parent.parent


@st.cache_resource
def get_database():
    """Reuse one MongoDB connection across dashboard reruns."""
    config = dotenv_values(PROJECT / "docker/mongodb.env")
    user = config.get("MONGO_INITDB_ROOT_USERNAME")
    password = config.get("MONGO_INITDB_ROOT_PASSWORD")
    if not user or not password:
        raise RuntimeError("Missing MongoDB credentials in docker/mongodb.env")

    uri = (
        f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
        "@localhost:27017/?authSource=admin"
    )
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client["gpuwaste"]


@st.cache_data(ttl=180, show_spinner=False)
def read_collection(name):
    """Our MongoDB collections contain small aggregated results."""
    documents = list(get_database()[name].find({}, {"_id": 0}))
    return pd.DataFrame(documents)


def as_numeric(df, column):
    """Safely convert a numeric column; missing columns return empty data."""
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def mean_label(df, column, unit=""):
    values = as_numeric(df, column).dropna()
    return f"{values.mean():.1f}{unit}" if not values.empty else "N/A"


def seren_dashboard():
    st.title("⚡ GPUWaste | Seren")
    st.caption("AcmeTrace — Seren production GPU cluster | Historical analytics")

    util = read_collection("seren_gpu_utilization")
    power = read_collection("seren_gpu_power")
    hourly = read_collection("seren_gpu_hourly")
    candidates = read_collection("seren_gpu_candidates")
    jobs = read_collection("seren_job_summary")

    if util.empty:
        st.warning("Seren data is missing. Run src/11_export_seren.py first.")
        return

    job = jobs.iloc[0] if not jobs.empty else {}
    tabs = st.tabs(["Overview", "Hourly trends", "GPU explorer", "Power & efficiency", "Jobs"])

    with tabs[0]:
        a, b, c, d = st.columns(4)
        a.metric("Monitored GPUs", f"{len(util):,}")
        b.metric("Mean GPU utilization*", mean_label(util, "avg_utilization", "%"))
        c.metric("Underutilization candidates", f"{len(candidates):,}")
        d.metric("Mean GPU power*", mean_label(power, "avg_power_w", " W"))

        st.subheader("Detection criteria")
        st.write(
            "A candidate must have average GPU utilization below 10%, "
            "at least 80% low-utilization observations, at least 80% valid "
            "monitoring coverage, and 1,000 or more valid samples."
        )
        if candidates.empty:
            st.info(
                "No Seren GPU meets all four criteria. This does not prove "
                "that Seren has no resource inefficiency."
            )
        st.caption("*Mean of per-GPU averages; not weighted by observation count.")

    with tabs[1]:
        st.subheader("Hourly GPU utilization")
        if {"hour", "avg_utilization"}.issubset(hourly.columns):
            chart = hourly.copy()
            chart["hour"] = pd.to_datetime(chart["hour"], errors="coerce")
            chart["avg_utilization"] = as_numeric(chart, "avg_utilization")
            chart = chart.dropna(subset=["hour", "avg_utilization"]).sort_values("hour")
            st.line_chart(chart.set_index("hour")["avg_utilization"])
            if "low_utilization_pct" in chart.columns:
                st.subheader("Share of low-utilization observations by hour")
                chart["low_utilization_pct"] = as_numeric(chart, "low_utilization_pct")
                st.line_chart(chart.set_index("hour")["low_utilization_pct"])
            st.dataframe(chart.tail(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Hourly summary is missing expected columns.")

    with tabs[2]:
        st.subheader("GPU explorer")
        if "gpu_id" in util.columns:
            gpu_ids = sorted(util["gpu_id"].dropna().astype(str).unique())
            gpu_id = st.selectbox("Select a Seren GPU", gpu_ids)
            selected = util[util["gpu_id"] == gpu_id].iloc[0]
            selected_power = power[power["gpu_id"] == gpu_id] if "gpu_id" in power else pd.DataFrame()
            a, b, c = st.columns(3)
            a.metric("Mean utilization", f"{float(selected.get('avg_utilization', 0)):.1f}%")
            b.metric("Low-utilization observations", f"{float(selected.get('low_utilization_pct', 0)):.1f}%")
            if not selected_power.empty:
                p = pd.to_numeric(pd.Series([selected_power.iloc[0].get("avg_power_w")]), errors="coerce").iloc[0]
                c.metric("Mean power", f"{p:.1f} W" if pd.notna(p) else "N/A")
            else:
                c.metric("Mean power", "N/A")

        st.subheader("Lowest average utilization GPUs")
        if "avg_utilization" in util:
            view = util.copy()
            view["avg_utilization"] = as_numeric(view, "avg_utilization")
            cols = [col for col in ["gpu_id", "avg_utilization", "low_utilization_pct", "coverage_pct", "valid_count"] if col in view]
            st.dataframe(view.sort_values("avg_utilization")[cols].head(50), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("GPU utilization vs. average power")
        if {"gpu_id", "avg_utilization"}.issubset(util.columns) and {"gpu_id", "avg_power_w"}.issubset(power.columns):
            merged = util[["gpu_id", "avg_utilization"]].merge(
                power[["gpu_id", "avg_power_w"]], on="gpu_id", how="inner"
            )
            merged["avg_utilization"] = as_numeric(merged, "avg_utilization")
            merged["avg_power_w"] = as_numeric(merged, "avg_power_w")
            merged = merged.dropna()
            st.scatter_chart(merged, x="avg_utilization", y="avg_power_w")
            st.caption(
                "Each point represents one GPU. These are historical averages, "
                "not timestamp-aligned measurements; no energy savings are inferred."
            )
        else:
            st.warning("Power or utilization fields are missing.")
        st.metric("Candidates meeting all criteria", len(candidates))
        if candidates.empty:
            st.info("No candidate-level efficiency records are expected for Seren.")

    with tabs[4]:
        st.subheader("Seren workload statistics")
        if not jobs.empty:
            a, b, c = st.columns(3)
            a.metric("Total jobs", f"{int(job.get('total_jobs', 0)):,}")
            b.metric("GPU-requesting jobs", f"{int(job.get('gpu_jobs', 0)):,}")
            c.metric("Sum of GPU requests", f"{int(job.get('requested_gpu_sum', 0)):,}")
            st.caption(
                "Sum of GPU requests counts requests across jobs, "
                "not the number of distinct physical GPUs."
            )
        else:
            st.warning("Seren job summary is missing.")


def compare_clusters():
    st.title("⚖️ GPUWaste | Kalos vs. Seren")
    st.caption("Comparison of two production clusters using the same candidate criteria")

    rows = []
    for label, prefix in [("Kalos", ""), ("Seren", "seren_")]:
        util = read_collection(prefix + "gpu_utilization")
        candidates = read_collection(prefix + "gpu_candidates")
        power = read_collection(prefix + "gpu_power")
        rows.append({
            "Cluster": label,
            "Monitored GPUs": len(util),
            "Candidates": len(candidates),
            "Candidate share (%)": round(100 * len(candidates) / len(util), 2) if len(util) else 0,
            "Mean of GPU avg utilization (%)": round(as_numeric(util, "avg_utilization").mean(), 2),
            "Mean of GPU avg power (W)": round(as_numeric(power, "avg_power_w").mean(), 2),
        })

    result = pd.DataFrame(rows)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.subheader("Number of monitored GPUs")
    st.bar_chart(result.set_index("Cluster")[["Monitored GPUs"]])
    st.subheader("Underutilization candidates")
    st.bar_chart(result.set_index("Cluster")[["Candidates"]])
    st.info(
        "196 Kalos candidates and 0 Seren candidates reflect the chosen detection "
        "rules. The clusters cover different workloads and time periods; "
        "these figures do not establish causation or energy savings."
    )


# Existing Kalos dashboard remains untouched.
st.set_page_config(page_title="GPUWaste — Kalos & Seren", page_icon="⚡", layout="wide")

page = st.navigation(
    [
        st.Page("09_dashboard.py", title="Kalos dashboard", icon="🟦", default=True),
        st.Page(seren_dashboard, title="Seren dashboard", icon="🟩"),
        st.Page(compare_clusters, title="Compare clusters", icon="⚖️"),
    ],
    position="sidebar",
)
page.run()
