import polars as pl
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import re

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

header_card = dbc.Card([
    dbc.CardHeader("Market Data Analyzer"),
])

text_1 = dbc.Card([dbc.CardHeader("Market Data"),
                   dbc.CardBody(
                       html.Pre(id='hover-output'),
                       style={"height": "110px"}
)])
text_2 = dbc.Card(
    [
        dbc.CardHeader("Market Prints"),
        dbc.CardBody()
    ],
    style={"height": "40%"},
)

text_3 = dbc.Card(
    [
        dbc.CardHeader("More Panels..."),
        dbc.CardBody()
    ],
    style={"height": "40%"},
)

market_data_card = dbc.Card(
    [
        dbc.CardHeader(["Market Data Panel  ",
                        dcc.Input(id="symbol-input", type="text",
                                  value="AAPL", placeholder="Enter Symbol"),
                        dcc.Input(id="date-input", type="text",
                                  value="20251002", placeholder="Select Date"),
                        dbc.Button("Load Data", id="load-btn", n_clicks=0),]),
        dbc.CardBody(
            dcc.Graph(
                id="market-data-panel",
                style={"height": "100%", "width": "100%"},
                config={
                    "responsive": True,
                    "displayModeBar": False,
                    "scrollZoom": True,
                },
            ),
            style={"height": "100%", "padding": 0},
        ),
    ],
    style={"height": "85%"},
)

current_pos_card = dbc.Card(
    [
        dbc.CardHeader("Current Position"),
        dbc.CardBody(),
    ],
    style={"height": "20%"},
)

app.layout = dbc.Container(
    [
        dcc.Store(id="nbbo-store"),
        dbc.Row([dbc.Col(header_card)], className="mb-2"),
        dbc.Row(
            [
                dbc.Col([text_1, text_2, text_3], width=2),
                dbc.Col([market_data_card, current_pos_card], width=True),
            ], class_name="mb-2",
            style={"height": "calc(100vh - 90px)"}
        ),
    ],
    fluid=True,
    style={"height": "100vh"}
)

# ----------------------------------------------------------------------


def load_data_from_date_and_symbol(date, symbol):
    match = re.match(r"(\d{4})(\d{2})(\d{2})", date)
    if not match:
        raise ValueError("Date must be in YYYYMMDD format")

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))

    start_time = datetime(year, month, day, 9, 30, 0)
    end_time = datetime(year, month, day, 16, 0, 0)

    execs = pl.read_parquet(
        f"data/execs_{date}.parquet").filter(pl.col("sym") == symbol)

    orders = pl.read_parquet(f"data/orders_{date}.parquet").filter(
        (pl.col("sym") == symbol) &
        (pl.col("place_date_time") >= start_time) &
        (pl.col("oc_date_time") <= end_time)
    )

    ref = pl.read_parquet(
        f"data/ref_data_{date}.parquet").filter(pl.col("sym") == symbol)

    nbbo = pl.read_parquet(f"data/US_nbbo_{date}.parquet").filter(
        (pl.col("sym") == symbol) &
        (pl.col("date_time") >= start_time) &
        (pl.col("date_time") <= end_time)
    ).drop(["sym"])

    nbbo = nbbo.with_columns(
        (pl.col("bid") / 10000).alias("bid"),
        (pl.col("ask") / 10000).alias("ask"),
        pl.col("date_time").dt.strftime("%H:%M:%S.%3f").alias("time_str"),
    )

    return execs, orders, ref, nbbo


def create_figure(nbbo, orders):
    fig_spread = px.line(
        nbbo,
        x="date_time",
        y=["bid", "ask"],
        color_discrete_sequence=["green", "red"],
    )
    fig_orders = px.scatter(
        orders,
        x="place_date_time",
        y="oprice",
        color_discrete_sequence=["black"],
    )

    fig = go.Figure(data=fig_spread.data + fig_orders.data)

    fig.update_layout(
        margin=dict(l=5, r=5, t=10, b=5),
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_thickness=0.08,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=15, label="15s",
                         step="second", stepmode="backward"),
                    dict(count=30, label="30s",
                         step="second", stepmode="backward"),
                    dict(count=1, label="1m", step="minute", stepmode="backward"),
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=15, label="15m",
                         step="minute", stepmode="backward"),
                    dict(step="all"),
                ],
                x=0.04,
                y=0.99,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return fig


def extract_xrange(relayout):
    if not relayout:
        return None, None

    xr = relayout.get("xaxis.range")
    if isinstance(xr, (list, tuple)) and len(xr) == 2:
        return xr[0], xr[1]

    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        return relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]

    # Sometimes plotly uses xaxis.rangeslider.range
    rs = relayout.get("xaxis.rangeslider.range")
    if isinstance(rs, (list, tuple)) and len(rs) == 2:
        return rs[0], rs[1]

    return None, None


@app.callback(
    Output("market-data-panel", "figure"),
    Output("nbbo-store", "data"),
    Input("load-btn", "n_clicks"),
    Input("market-data-panel", "relayoutData"),
    State("date-input", "value"),
    State("symbol-input", "value"),
    State("nbbo-store", "data"),
    State("market-data-panel", "figure"),
    prevent_initial_call=False,
)
def update_chart(n_clicks, relayout, date_str, symbol, nbbo_store, fig):
    trigger = ctx.triggered_id

    # 1) Load data
    if trigger == "load-btn":
        symbol = (symbol or "").strip().upper()
        date = (date_str or "").replace("-", "")

        execs, orders, ref, nbbo = load_data_from_date_and_symbol(date, symbol)

        # Store from polars without pandas/pyarrow
        nbbo_small = nbbo.select(["date_time", "bid", "ask"])
        nbbo_payload = {
            "date_time": [str(x) for x in nbbo_small.get_column("date_time").to_list()],
            "bid": [float(x) for x in nbbo_small.get_column("bid").to_list()],
            "ask": [float(x) for x in nbbo_small.get_column("ask").to_list()],
        }

        new_fig = create_figure(nbbo, orders)
        return new_fig, nbbo_payload

    # 2) Zoom/slider -> rescale y using stored NBBO
    if trigger == "market-data-panel":
        if not relayout or not nbbo_store or not fig:
            raise PreventUpdate

        x0, x1 = extract_xrange(relayout)
        if x0 is None or x1 is None:
            raise PreventUpdate

        x0_dt = pd.to_datetime(x0, errors="coerce")
        x1_dt = pd.to_datetime(x1, errors="coerce")
        if pd.isna(x0_dt) or pd.isna(x1_dt):
            raise PreventUpdate

        dt = pd.to_datetime(nbbo_store["date_time"], errors="coerce")
        bid = pd.Series(nbbo_store["bid"], dtype="float64")
        ask = pd.Series(nbbo_store["ask"], dtype="float64")

        mask = (dt >= x0_dt) & (dt <= x1_dt)
        if not mask.any():
            raise PreventUpdate

        ymin = float(min(bid[mask].min(), ask[mask].min()))
        ymax = float(max(bid[mask].max(), ask[mask].max()))

        pad = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
        yrange = [ymin - pad, ymax + pad]

        layout = fig.setdefault("layout", {})
        layout.setdefault("xaxis", {})["range"] = [x0, x1]
        layout["xaxis"]["autorange"] = False

        layout.setdefault("yaxis", {})["range"] = yrange
        layout["yaxis"]["autorange"] = False

        return fig, no_update

    raise PreventUpdate


@app.callback(  # -- Update Market Data Panel --
    Output("hover-output", "children"),
    Input("market-data-panel", "hoverData"),
)
def display_hover_data(hoverData):
    if hoverData is None:
        return f"Best Ask: -\nBest Bid: -\nLast Trade: -\nTime: -\n"

    summary = json.dumps(hoverData, indent=2)
    points = json.loads(summary)['points']

    marks = {'Best Ask': round(points[0]['y'], 2),
             'Best Bid': round(points[1]['y'], 2),
             'Last Trade': round(points[2]['y'], 2) if len(points) > 2 else '-',
             'Time': points[0]['x'].split(" ")[1],
             }

    return [f"{k}: {v}\n" for (k, v) in marks.items()]


if __name__ == "__main__":
    app.run()
