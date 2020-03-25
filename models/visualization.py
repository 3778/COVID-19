import numpy as np
import pandas as pd
import altair as alt


# ==========================================================
#                  SEIR-BAYES CHART
# ==========================================================

plot_params = {
    "exposed": {"name": "Exposed", "color": "#1f77b4"},
    "infected": {"name": "Infected", "color": "#ff7f0e"},
}


# enable tooltips
def tooltips():
    return {"config": {"mark": {"tooltip": {"content": "encoding"}}}}


alt.themes.register("tooltips", tooltips)
alt.themes.enable("tooltips")


def unstack_iterations_ndarray(arr: np.ndarray, t_space: np.array, name: str):
    return (
        pd.DataFrame(arr, index=t_space)
        .unstack()
        .reset_index()
        .rename(columns={"level_0": "iteration", "level_1": "day", 0: name})
    )


def droplevel_col_index(df: pd.DataFrame):
    df.columns = df.columns.droplevel()
    return df


def compute_mean_and_boundaries(df: pd.DataFrame, variable: str):
    if variable not in df.columns:
        raise ValueError(
            f"No column named {variable}. Variable name must refer to a dataframe column"
        )

    return (
        df.groupby("day")
        .agg({variable: [np.mean, np.std]})
        .pipe(droplevel_col_index)
        .assign(upper=lambda df: df["mean"] + df["std"])
        .assign(lower=lambda df: df["mean"] - df["std"])
        .add_prefix(variable + "_")
    )


def prep_tidy_data_to_plot(E, I, t_space):
    df_E = unstack_iterations_ndarray(E, t_space, plot_params["exposed"]["name"])
    df_I = unstack_iterations_ndarray(I, t_space, plot_params["infected"]["name"])

    agg_df_E = compute_mean_and_boundaries(df_E, plot_params["exposed"]["name"])
    agg_df_I = compute_mean_and_boundaries(df_I, plot_params["infected"]["name"])

    source = agg_df_E.merge(
        agg_df_I, how="left", left_index=True, right_index=True, validate="1:1"
    ).reset_index()
    return source


def make_exposed_infected_line_chart(source: pd.DataFrame, scale="log"):
    return (
        alt.Chart(
            source,
            width=800,
            height=500,
            title="Evolução no tempo de pessoas expostas e infectadas pelo COVID-19",
        )
        .transform_fold(
            ["Exposed_mean", "Infected_mean"],
            ["Variável", "Valor"],  # equivalent to id_vars in pandas' melt
        )
        .mark_line()
        .encode(
            x=alt.X("day:Q", title="Dias"),
            y=alt.Y("Valor:Q", title="Qtde. de pessoas", scale=alt.Scale(type=scale)),
            color="Variável:N",
        )
    )


def _treat_negative_values_to_plot(df):
    df[df <= 0] = 1.0
    return df


def make_exposed_infected_error_area_chart(
    source: pd.DataFrame, variable: str, color: str, scale: str = "log"
):
    treated_source = _treat_negative_values_to_plot(source)
    return (
        alt.Chart(treated_source)
        .transform_filter(f"datum.{variable}_lower > 0")
        .mark_area(color=color)
        .encode(
            x=alt.X("day:Q"),
            y=alt.Y(f"{variable}_upper", scale=alt.Scale(type=scale)),
            y2=f"{variable}_lower",
            opacity=alt.value(0.2),
        )
    )


def make_combined_chart(source, scale="log", show_uncertainty=True):
    lines = make_exposed_infected_line_chart(source, scale=scale)

    if not show_uncertainty:
        output = alt.layer(lines)

    else:
        band_E = make_exposed_infected_error_area_chart(
            source,
            plot_params["exposed"]["name"],
            plot_params["exposed"]["color"],
            scale=scale,
        )
        band_I = make_exposed_infected_error_area_chart(
            source,
            plot_params["infected"]["name"],
            plot_params["infected"]["color"],
            scale=scale,
        )
        output = alt.layer(band_E, band_I, lines)

    return (
        output.configure_title(fontSize=16)
        .configure_axis(labelFontSize=14, titleFontSize=14)
        .configure_legend(labelFontSize=14, titleFontSize=14)
        .interactive()
    )


# ==========================================================
#                QUEUE SIMULATION CHARTS
# ==========================================================

# TODO: use the same parameters used in simulation
MAX_NORMAL_BEDS = 200
MAX_UTI_BEDS = 30

# Changes bar color to red when occupancy > threshold*capacity
OCCUPANCY_THRESHOLD_ALERT = 0.80

# In case of layouting issues, you might adjust spacing 
# and padding parameters in `combine_queue_charts`
CHART_HEIGHT = 150
BAR_CHART_WIDHT = 50
LINE_CHART_WIDHT = 250


def prep_queue_data_to_plot(
    model_audit_report: pd.DataFrame, max_occuppied_normal_beds, max_occupied_icu_beds
):
    source_wide = (
        model_audit_report.loc[
            :, ["Time", "Queue", "ICU_Queue", "Occupied_beds", "ICU_Occupied_beds"]
        ]
        .assign(max_Occupied_beds=max_occuppied_normal_beds)
        .assign(max_ICU_Occupied_beds=max_occupied_icu_beds)
    )

    capacity = source_wide[["Time", "max_Occupied_beds", "max_ICU_Occupied_beds"]]
    capacity_melt = (
        capacity.melt(id_vars=["Time"], var_name="Kind", value_name="Capacity")
        .replace("max_Occupied_beds", "Normal")
        .replace("max_ICU_Occupied_beds", "UTI")
    )
    occupancy = source_wide[["Time", "Occupied_beds", "ICU_Occupied_beds"]]
    occupancy_melt = (
        occupancy.melt(id_vars=["Time"], var_name="Kind", value_name="Occupancy")
        .replace("Occupied_beds", "Normal")
        .replace("ICU_Occupied_beds", "UTI")
    )

    source_long = capacity_melt.merge(occupancy_melt, on=["Time", "Kind"])

    return source_wide, source_long


def make_queue_line_chart(source, queue_col_name, queue_sim_time):
    """
    bed_type_name: 'Queue' or 'ICU_Queue'
    """
    return (
        alt.Chart(source, title="")
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("Time", title="Dias"), y=alt.Y(f"{queue_col_name}:Q", title="Fila")
        )
        .transform_filter(alt.datum.Time <= queue_sim_time)
        .properties(width=LINE_CHART_WIDHT, height=CHART_HEIGHT)
    )


def make_queue_bar_with_tick_chart(source, title, bed_type_name, queue_sim_time):
    """
    bed_type_name: 'Normal' or 'UTI'
    """
    bar = (
        alt.Chart(source, title=title)
        .mark_bar()
        .encode(
            x=alt.X("Kind", axis=alt.Axis(labels=False), title=""),
            y=alt.Y("Occupancy", title=""),
            color=alt.condition(
                alt.datum.Occupancy > OCCUPANCY_THRESHOLD_ALERT * alt.datum.Capacity,
                alt.value("#D2222D"),  # not so bright red
                alt.value("steelblue"),
            ),
        )
        .transform_filter(
            (alt.datum.Time == queue_sim_time) & (alt.datum.Kind == bed_type_name)
        )
        .properties(width=BAR_CHART_WIDHT, height=CHART_HEIGHT)
    )

    tick = (
        alt.Chart(source)
        .mark_tick(
            color="red",
            thickness=2,
            size=BAR_CHART_WIDHT * 0.9,  # controls width of tick
        )
        .encode(x="Kind", y="Capacity")
        .transform_filter(
            (alt.datum.Time == queue_sim_time) & (alt.datum.Kind == bed_type_name)
        )
        .properties(width=BAR_CHART_WIDHT, height=CHART_HEIGHT)
    )

    return bar, tick


def make_queue_occupancy_line_chart(source, bed_type_name, queue_sim_time):
    return (
        alt.Chart(source)
        .transform_fold(["Capacity", "Occupancy"])
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("Time", title="", axis=alt.Axis(labels=False)),
            y=alt.Y("value:Q", title="Ocupação"),
            color=alt.Color(
                "key:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Capacity", "Occupancy"], range=["red", "steelblue"]
                ),
            ),
        )
        .transform_filter(
            (alt.datum.Time <= queue_sim_time) & (alt.datum.Kind == bed_type_name)
        )
        .properties(width=LINE_CHART_WIDHT, height=CHART_HEIGHT)
    )


def make_queue_charts(source_long, source_wide, queue_sim_time):
    line_queue_normal = make_queue_line_chart(source_wide, "Queue", queue_sim_time)
    line_queue_uti = make_queue_line_chart(source_wide, "ICU_Queue", queue_sim_time)
    bar_normal, tick_normal = make_queue_bar_with_tick_chart(
        source_long, "Leitos normais", "Normal", queue_sim_time
    )
    bar_uti, tick_uti = make_queue_bar_with_tick_chart(
        source_long, "Leitos UTI", "UTI", queue_sim_time
    )
    line_normal = make_queue_occupancy_line_chart(source_long, "Normal", queue_sim_time)
    line_uti = make_queue_occupancy_line_chart(source_long, "UTI", queue_sim_time)

    charts = {
        "normal": {
            "bar": bar_normal,
            "tick": tick_normal,
            "occupancy": line_normal,
            "queue": line_queue_normal,
        },
        "uti": {
            "bar": bar_uti,
            "tick": tick_uti,
            "occupancy": line_uti,
            "queue": line_queue_uti,
        },
    }

    return charts


def combine_queue_charts(charts):
    cap_occ_bar_chart_normal = charts["normal"]["bar"] + charts["normal"]["tick"]
    cap_occ_bar_chart_uti = charts["uti"]["bar"] + charts["uti"]["tick"]
    return alt.vconcat(
        alt.hconcat(
            alt.vconcat(
                cap_occ_bar_chart_normal,
                charts["normal"]["occupancy"],
                charts["normal"]["queue"],
                center=True,
            ),
            alt.vconcat(
                cap_occ_bar_chart_uti,
                charts["uti"]["occupancy"],
                charts["uti"]["queue"],
                center=True,
            ),
            spacing=100,
        ),
        center=True,
        spacing=50,
        padding={"left": 50, "top": 10},
    )


def make_queue_charts_wrapper(model_audit_report, queue_sim_time):
    source_wide, source_long = prep_queue_data_to_plot(
        model_audit_report, MAX_NORMAL_BEDS, MAX_UTI_BEDS
    )
    charts = make_queue_charts(source_long, source_wide, queue_sim_time)
    return combine_queue_charts(charts)
