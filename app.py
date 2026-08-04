from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from shiny import App, reactive, render, ui


DATA_PATH = Path(__file__).with_name("bakerysales.csv")
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

sales_data = pd.read_csv(DATA_PATH, parse_dates=["date"])
train = sales_data[sales_data["date"].dt.year <= 2016].copy()
test = sales_data[sales_data["date"].dt.year == 2017].copy()

# Management's current promotion-only model.
baseline_model = smf.ols("sales ~ promotion", data=train).fit()

# Recommended model: log transformation + promotion, weather, store, and weekday.
log_model = smf.ols(
    "np.log1p(sales) ~ promotion + bad_weather + C(store_type) + C(day)",
    data=train,
).fit()

# Convert log predictions back into units sold using a smearing correction.
smearing_factor = np.mean(np.exp(log_model.resid))
test["baseline_prediction"] = baseline_model.predict(test)
test["log_model_prediction"] = np.maximum(
    0, np.exp(log_model.predict(test)) * smearing_factor - 1
)
train["log_fitted"] = log_model.fittedvalues
train["log_residual"] = log_model.resid


app_ui = ui.page_fluid(
    ui.h2("Bakery sales forecasting dashboard"),
    ui.p(
        "2017 forecasts for one bakery product. The recommended model uses a "
        "log-sales transformation, promotion, weather, store type, and day of week."
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Forecast dashboard",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "store_id",
                        "Store",
                        choices={str(i): f"Store {i}" for i in range(1, 11)},
                        selected="1",
                    ),
                    ui.input_slider(
                        "month_range",
                        "2017 months",
                        min=1,
                        max=12,
                        value=(1, 12),
                        step=1,
                        sep="",
                    ),
                    ui.p("Select one month by setting both slider handles to the same month."),
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Actual sales vs. manager's promotion-only forecast"),
                        ui.output_plot("baseline_plot"),
                    ),
                    ui.card(
                        ui.card_header("Actual sales vs. recommended log-sales forecast"),
                        ui.output_plot("recommended_plot"),
                    ),
                    col_widths=(6, 6),
                ),
                ui.card(
                    ui.card_header("Forecast accuracy for the selected store and period"),
                    ui.output_data_frame("accuracy_table"),
                    ui.p(
                        "RMSE (Root Mean Squared Error) gives extra weight to large "
                        "mistakes. MAE (Mean Absolute Error) is the typical size of a "
                        "mistake. Lower values are better."
                    ),
                ),
            ),
        ),
        ui.nav_panel(
            "Model checks",
            ui.p(
                "These checks use the 2013-2016 training data. The log transformation "
                "makes the residuals more balanced than the raw-sales model."
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Residual distribution"), ui.output_plot("residual_hist")),
                ui.card(ui.card_header("Residuals versus fitted values"), ui.output_plot("residual_plot")),
                col_widths=(6, 6),
            ),
            ui.p(
                "A residual is actual sales minus predicted sales. A balanced cloud around "
                "zero suggests the model does not have a consistent prediction bias."
            ),
        ),
    ),
)


def server(input, output, session):
    @reactive.calc
    def selected_data():
        start_month, end_month = input.month_range()
        return test[
            (test["store_id"] == int(input.store_id()))
            & (test["date"].dt.month >= start_month)
            & (test["date"].dt.month <= end_month)
        ].sort_values("date")

    def draw_forecast(selected, prediction_column, prediction_label, prediction_color, title_prefix):
        """Draw a two-line comparison on a shared scale for clear side-by-side review."""
        values = pd.concat(
            [
                selected["sales"],
                selected["baseline_prediction"],
                selected["log_model_prediction"],
            ]
        )
        padding = (values.max() - values.min()) * 0.05
        y_min = max(0, values.min() - padding)
        y_max = values.max() + padding
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(selected["date"], selected["sales"], color="black", linewidth=1.8, label="Actual sales")
        ax.plot(selected["date"], selected[prediction_column], color=prediction_color, label=prediction_label)
        ax.set_title(title_prefix)
        ax.set_xlabel("Date")
        ax.set_ylabel("Units sold")
        ax.set_ylim(y_min, y_max)
        ax.legend()
        fig.tight_layout()
        return fig

    @render.plot
    def baseline_plot():
        selected = selected_data()
        start_month, end_month = input.month_range()
        return draw_forecast(
            selected,
            "baseline_prediction",
            "Promotion-only forecast",
            "#4c78a8",
            f"Store {input.store_id()}: {MONTHS[start_month - 1]} to {MONTHS[end_month - 1]}, 2017",
        )

    @render.plot
    def recommended_plot():
        selected = selected_data()
        start_month, end_month = input.month_range()
        return draw_forecast(
            selected,
            "log_model_prediction",
            "Recommended log-sales forecast",
            "#f58518",
            f"Store {input.store_id()}: {MONTHS[start_month - 1]} to {MONTHS[end_month - 1]}, 2017",
        )

    @render.data_frame
    def accuracy_table():
        selected = selected_data()
        baseline_error = selected["sales"] - selected["baseline_prediction"]
        log_error = selected["sales"] - selected["log_model_prediction"]
        metrics = pd.DataFrame(
            {
                "Model": ["Promotion only", "Recommended log-sales model"],
                "RMSE": [
                    np.sqrt(np.mean(baseline_error**2)),
                    np.sqrt(np.mean(log_error**2)),
                ],
                "MAE": [
                    np.mean(np.abs(baseline_error)),
                    np.mean(np.abs(log_error)),
                ],
            }
        ).round(2)
        return render.DataGrid(metrics, summary=False)

    @render.plot
    def residual_hist():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(train["log_residual"], bins=40, color="#4c78a8", edgecolor="white")
        ax.axvline(0, color="#e45756", linestyle="--")
        ax.set_xlabel("Residual (actual log sales minus predicted log sales)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    @render.plot
    def residual_plot():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(train["log_fitted"], train["log_residual"], alpha=0.22, s=12, color="#4c78a8")
        ax.axhline(0, color="#e45756", linestyle="--")
        ax.set_xlabel("Predicted log sales")
        ax.set_ylabel("Residual")
        fig.tight_layout()
        return fig


app = App(app_ui, server)
