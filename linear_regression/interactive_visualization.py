import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import utils


def plot_grid_search(x, y, slopes, prediction, loss, dloss_dw):
    pred_df = pd.DataFrame(prediction)
    loss_df = pd.DataFrame({"slope": slopes, "loss": loss})
    dloss_dw_df = pd.DataFrame({"slope": slopes, "dloss/dw": dloss_dw})

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Data & Fitted Line", "Loss", "dLoss/dw")
    )

    # fig 1
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10),
            name="Data",
            line_color="#5D5DA6",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pred_df.iloc[:, 0],
            line_color="#F9B041",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(row=1, col=1, title="feature")
    fig.update_yaxes(row=1, col=1, title="target")

    # fig 2
    fig.add_trace(
        go.Scatter(
            x=loss_df["slope"],
            y=loss_df["loss"],
            mode="markers",
            marker=dict(size=7),
            name="Loss",
            line_color="#2DA9E1",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=loss_df.iloc[[0]]["slope"],
            y=loss_df.iloc[[0]]["loss"],
            line_color="red",
            mode="markers",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            name="Loss for line",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(row=1, col=2, title="w")

    # fig 3
    fig.add_trace(
        go.Scatter(
            x=dloss_dw_df["slope"],
            y=dloss_dw_df["dloss/dw"],
            mode="markers",
            marker=dict(size=7),
            name="derivative",
            line_color="#4AAE4D",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=dloss_dw_df.iloc[[0]]["slope"],
            y=dloss_dw_df.iloc[[0]]["dloss/dw"],
            line_color="yellow",
            mode="markers",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            name="derivative for Loss",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(row=1, col=3, title="w")

    # movement
    frames = [
        dict(
            name=f"{slope}",
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=pred_df[f"{slope}"]),
                go.Scatter(x=loss_df["slope"], y=loss_df["loss"]),
                go.Scatter(
                    x=loss_df.iloc[[n]]["slope"],
                    y=loss_df.iloc[[n]]["loss"],
                ),
                go.Scatter(x=dloss_dw_df["slope"], y=dloss_dw_df["dloss/dw"]),
                go.Scatter(
                    x=dloss_dw_df.iloc[[n]]["slope"],
                    y=dloss_dw_df.iloc[[n]]["dloss/dw"],
                ),
            ],
            traces=[0, 1, 2, 3, 4, 5],
        )
        for n, slope in enumerate(slopes)
    ]

    # slider
    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "w: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{slope}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{slope}",
                    "method": "animate",
                }
                for slope in slopes
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_gradient_descent(
    x, y, slopes, prediction, mses, dmse_dws, w_range=(2.5, 17.5, 0.05)
):
    pred_df = pd.DataFrame(prediction)
    slope_range = np.arange(*w_range)
    mse = []
    for w in slope_range:
        mse.append(mean_squared_error(y, w * x + 2.83))  # calc MSE
    mse = pd.DataFrame({"slope": slope_range, "squared_error": mse})
    iters = np.arange(len(dmse_dws)) + 1

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Data & Fitted Line", "Mean Squared Error", "dMSE/dw"),
    )

    # fig 1
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=10),
            name="Data",
            line_color="#5D5DA6",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pred_df.iloc[:, 0],
            line_color="#F9B041",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[2],
            y=[95],
            mode="text",
            text=f"<b>w = {slopes[0]:.2f}<b>",
            textfont=dict(size=16, color="DarkSlateGrey"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(row=1, col=1, title="feature")
    fig.update_yaxes(row=1, col=1, title="target")

    # fig 2
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            line_color="#2DA9E1",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[:1]),
            y=np.array(mses[:1]),
            line_color="salmon",
            line=dict(width=4),
            mode="markers+lines",
            name="Slope history",
            marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[0]),
            y=np.array(mses[0]),
            line_color="red",
            mode="markers",
            name="MSE for line",
            marker=dict(size=18, line=dict(width=1, color="DarkSlateGrey")),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(row=1, col=2, title="w")

    # fig 3
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=dmse_dws,
            mode="markers",
            line_color="DarkSlateGrey",
            marker=dict(size=3),
            name="Gradient values",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(iters[0]),
            y=np.array(dmse_dws[0]),
            mode="markers",
            marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
            name="Gradient history",
            line_color="#4AAE4D",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(row=1, col=3, title="iteration")
    # movement
    frames = [
        dict(
            name=n,
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=pred_df[slope]),
                go.Scatter(text=f"<b>w = {slope:.2f}<b>"),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(
                    x=np.array(slopes[: n + 1]),
                    y=np.array(mses[: n + 1]),
                    mode="markers" if n == 0 else "markers+lines",
                ),
                go.Scatter(x=np.array(slopes[n]), y=np.array(mses[n])),
                go.Scatter(x=iters, y=dmse_dws),
                go.Scatter(
                    x=np.array(iters[: n + 1]),
                    y=np.array(dmse_dws[: n + 1]),
                    mode="markers" if n == 0 else "markers+lines",
                ),
            ],
            traces=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        for n, slope in enumerate(slopes)
    ]
    # slider
    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Iteration: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [n],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": n,
                    "method": "animate",
                }
                for n in range(len(slopes))
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_grid_search_2d(x, y, slopes, intercepts):
    mse = np.zeros((len(slopes), len(intercepts)))
    for i, slope in enumerate(slopes):
        for j, intercept in enumerate(intercepts):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Surface Plot", "Contour Plot"),
        specs=[[{"type": "surface"}, {"type": "contour"}]],
    )
    # fig 1
    fig.add_trace(
        go.Surface(z=mse, x=intercepts, y=slopes, name="", colorscale="RdYlGn_r"),
        row=1,
        col=1,
    )
    # fig 2
    fig.add_trace(
        go.Contour(
            z=mse,
            x=intercepts,
            y=slopes,
            name="",
            showscale=False,
            colorscale="RdYlGn_r",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            zaxis=dict(title="MSE"),
            yaxis=dict(title="slope (w)"),
            xaxis=dict(title="intercept (b)"),
        ),
        scene_camera=dict(eye=dict(x=2, y=1.1, z=1.2)),
        margin=dict(l=0, r=0, b=60, t=90),
    )
    fig.update_xaxes(
        title="intercept (b)",
        range=[intercepts.min(), intercepts.max()],
        tick0=intercepts.max(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w)",
        range=[slopes.min(), slopes.max()],
        tick0=slopes.min(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_layout(width=900, height=475, margin=dict(t=60))
    return fig


def plot_gradient_descent_2d(
    x, y, ws, slopes, intercepts, title="Gradient Descent", mode="markers+lines"
):
    bs, ws = ws[:, 0], ws[:, 1]
    mse = np.zeros((len(slopes), len(intercepts)))
    for i, slope in enumerate(slopes):
        for j, intercept in enumerate(intercepts):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)

    fig = make_subplots(
        rows=1,
        subplot_titles=[title],
    )
    fig.add_trace(
        go.Contour(
            z=mse,
            x=intercepts,
            y=slopes,
            name="",
            showscale=False,
            colorscale="RdYlGn_r",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bs,
            y=ws,
            mode=mode,
            line=dict(width=2.5),
            line_color="coral",
            marker=dict(
                opacity=1,
                size=np.linspace(19, 1, len(intercepts)),
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            name="Descent Path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[bs[0]],
            y=[ws[0]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="orangered",
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[bs[-1]],
            y=[ws[-1]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="yellowgreen",
            name="End",
        )
    )
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        title="intercept (b)",
        range=[intercepts.min(), intercepts.max()],
        tick0=intercepts.max(),
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w)",
        range=[slopes.min(), slopes.max()],
        tick0=slopes.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    return fig


def plot_panel(f1, f2, f3):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Gradient Descent",
            "Stochastic Gradient Descent",
            "Minibatch Gradient Descent",
        ),
    )
    for n, f in enumerate((f1, f2, f3)):
        for _ in range(len(f.data)):
            fig.add_trace(f.data[_], row=1, col=n + 1)
            fig.update_xaxes(title="intercept (b)")
            fig.update_yaxes(title="slope (w)")
    fig.update_layout(width=1000, height=400, margin=dict(t=60), showlegend=False)
    return fig
