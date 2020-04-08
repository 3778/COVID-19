import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go


plot_params = {
    "exposed": {"name": "Exposed", "color": "#1f77b4"},
    "infected": {"name": "Infected", "color": "#ff7f0e"},
}

plotly_line_colors = {
    "official": "#1f77b4",
    "exposed" : "#1f77b4",
    "forecast": "#ff7f0e",
    "infected": "#ff7f0e"
}

plotly_default_layout = dict(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1a",
                     step="year",
                     stepmode="backward"),
                dict(step="all", label="tudo")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ),
    width=900,
    height=400, 
    yaxis_tickformat = ',',
    margin=dict(l=0, r=0, t=40, b=20)
)


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
        .rename(columns={"level_0": "iteration", "level_1": "Dias", 0: name})
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
        df.groupby("Dias")
        .agg({variable: [np.mean, np.std]})
        .pipe(droplevel_col_index)
        .assign(upper=lambda df: df["mean"] + df["std"])
        .assign(lower=lambda df: df["mean"] - df["std"])
        .add_prefix(variable + "_")
    )


def prep_tidy_data_to_plot(E, I, t_space, time_index):
    df_E = unstack_iterations_ndarray(E, t_space, plot_params["exposed"]["name"])
    df_I = unstack_iterations_ndarray(I, t_space, plot_params["infected"]["name"])

    agg_df_E = compute_mean_and_boundaries(df_E, plot_params["exposed"]["name"])
    agg_df_I = compute_mean_and_boundaries(df_I, plot_params["infected"]["name"])

    data = (
        agg_df_E
        .merge(
            agg_df_I,
            how="left",
            left_index=True,
            right_index=True,
            validate="1:1"
        ).reset_index()
    )
    return data.set_index(time_index)


def make_exposed_infected_line_chart(data: pd.DataFrame, scale="log"):
    return (
        alt.Chart(
            data.rename(columns={'Exposed_mean': 'Indivíduos expostos', 'Infected_mean': 'Indivíduos infectados'}),
            width=600,
            height=500,
            title="Evolução no tempo de pessoas expostas e infectadas pelo COVID-19",
        )
        .transform_fold(
            ["Indivíduos expostos", "Indivíduos infectados"],
            ["Variável", "Valor"]  # equivalent to id_vars in pandas' melt
        )

        .mark_line()
        .encode(
            x=alt.X("Dias:Q", title="Dias"),
            y=alt.Y("Valor:Q", title="Qtde. de pessoas", scale=alt.Scale(type=scale)),
            color="Variável:N",
        )
    )


def _treat_negative_values_to_plot(df):
    df[df <= 0] = 1.0
    return df


def make_exposed_infected_error_area_chart(
    data: pd.DataFrame, variable: str, color: str, scale: str = "log"
):
    treated_source = _treat_negative_values_to_plot(data)
    return (
        alt.Chart(treated_source)
        .transform_filter(f"datum.{variable}_lower > 0")
        .mark_area(color=color)
        .encode(
            x=alt.X("Dias:Q"),
            y=alt.Y(f"{variable}_upper", scale=alt.Scale(type=scale)),
            y2=f"{variable}_lower",
            opacity=alt.value(0.2),
        )
    )


def make_combined_chart(data, scale="log", show_uncertainty=True):
    lines = make_exposed_infected_line_chart(data, scale=scale)

    if not show_uncertainty:
        output = alt.layer(lines)

    else:
        band_E = make_exposed_infected_error_area_chart(
            data,
            plot_params["exposed"]["name"],
            plot_params["exposed"]["color"],
            scale=scale,
        )
        band_I = make_exposed_infected_error_area_chart(
            data,
            plot_params["infected"]["name"],
            plot_params["infected"]["color"],
            scale=scale,
        )
        output = alt.layer(band_E, band_I, lines)

    return (
        alt.vconcat(
            output.interactive(),
            padding={"top": 20}
        )
        .configure_title(fontSize=16)
        .configure_axis(labelFontSize=14, titleFontSize=14)
        .configure_legend(labelFontSize=14, titleFontSize=14)
    )


def plot_r0(r0_samples, date, place, min_days):
    r0_samples_cut = r0_samples[-min_days:]
    columns = pd.date_range(end=date, periods=r0_samples_cut.shape[1])
    data = (pd.DataFrame(r0_samples_cut, columns=columns)
              .stack(level=0)
              .reset_index()
              .rename(columns={'level_1': 'Dias',
                               0: 'r0'})
              [['Dias', 'r0']])
    line = (
        alt
        .Chart(
            data,
            width=600,
            height=300,
            title=f"Número básico de reprodução para {place}"
        )
        .mark_line()
        .encode(
            x='Dias',
            y='mean(r0)'
        )
    )

    band = alt.Chart(data).mark_errorband(extent='stdev').encode(
        x='Dias',
        y=alt.Y('r0', title='Valor'),
    )

    return band + line


# def make_plotly_plot_r0(r0_samples, date, place, min_days):
#     r0_samples_cut = r0_samples[-min_days:]
#     columns = pd.date_range(end=date, periods=r0_samples_cut.shape[1])
#     data = (pd.DataFrame(r0_samples_cut, columns=columns)
#               .stack(level=0)
#               .reset_index()
#               .rename(columns={'level_1': 'Dias',
#                                0: 'r0'})
#               [['Dias', 'r0']])
#     print(data)
#     y1_upper = r0_samples_cut.mean(axis=1) + r0_samples_cut.std(axis=1)
#     print(y1_upper)

def make_plotly_solo_chart(df):
    y = df.values
    x = df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines',
                        name=df.name,
                        showlegend=True,
                        marker=dict(color=plotly_line_colors['exposed'])
                        ))

    fig.update_layout(plotly_default_layout)
    
    return fig

def make_plotly_capacity_chart(df, capacity):
    y = df.values
    x = df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines',
                        name=df.name,
                        showlegend=True,
                        marker=dict(color=plotly_line_colors['exposed'])
                        ))

    fig.add_shape(
        # Line Horizontal
            type="line",
            x0=x[0],
            y0=capacity,
            x1=x[-1],
            y1=capacity,
            line=dict(
                color="red",
                width=1,
                dash="dashdot"
            ),
    )

    fig.add_trace(go.Scatter(x=[x[0]], y=[capacity],
                    mode='lines',
                    name='Capacidade',
                    showlegend=True,
                    marker=dict(color='red')))

    fig.update_layout(plotly_default_layout)
    
    return fig

def make_plotly_combined_chart(df, y1, y2):
    y = df.values
    x = df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df[y1],
                        mode='lines',
                        name=y1,
                        showlegend=True,
                        marker=dict(color=plotly_line_colors['exposed'])
                        ))

    fig.add_trace(go.Scatter(x=x, y=df[y2],
                        mode='lines',
                        name=y2,
                        showlegend=True,
                        marker=dict(color=plotly_line_colors['infected'])
                        ))

    fig.update_layout(plotly_default_layout)

    return fig
