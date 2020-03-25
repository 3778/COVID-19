import streamlit as st

import altair as alt
import pandas as pd

from models.visualization import make_queue_charts_wrapper


# TODO: integrate with simulation output (this is only a sample)
MODEL_AUDIT_REPORT_PATH = "data/csv/queue_simulation/model_audit_report.csv"

if __name__ == "__main__":

    st.markdown("## Queue simulation viz prototype")

    model_audit_report = pd.read_csv(MODEL_AUDIT_REPORT_PATH)

    st.text(f"Model audit report ({MODEL_AUDIT_REPORT_PATH})")
    st.dataframe(model_audit_report)
    st.markdown("---")

    min_time, max_time = (
        int(model_audit_report["Time"].min()),
        int(model_audit_report["Time"].max()),
    )
    queue_sim_time = st.slider(
        "Dia da simulação",
        value=min_time,
        min_value=min_time,
        max_value=max_time,
        step=1,
    )

    queue_combined_charts = make_queue_charts_wrapper(
        model_audit_report, queue_sim_time
    )
    st.write(queue_combined_charts)
