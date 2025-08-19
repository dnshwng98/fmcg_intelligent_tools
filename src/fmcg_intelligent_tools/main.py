import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

from constants import POLAND_PUBLIC_HOLIDAYS
from utils import chunk_date, categorize_weekend, map_season_month, map_public_holiday, \
    sarimax_grid_search, create_month_day_series, download_dataset


def main():
    st.write("# Demand Forecasting")
    st.write("---")
    st.write("## Initialization")
    dataset_url = st.text_input("Dataset URL", "beatafaron/fmcg-daily-sales-data-to-2022-2024", disabled=True)

    if "fmcg_dataset" not in st.session_state:
        st.session_state.fmcg_dataset = None

    if st.button("Download and Preview"):
        st.session_state.fmcg_dataset = download_dataset(dataset_url)

    # Feature engineering
    public_holidays_list = create_month_day_series(POLAND_PUBLIC_HOLIDAYS)

    if st.button("Add Feature Engineering"):
        if st.session_state.fmcg_dataset is not None:
            st.session_state.fmcg_dataset = chunk_date(st.session_state.fmcg_dataset)
            st.session_state.fmcg_dataset = categorize_weekend(st.session_state.fmcg_dataset)
            st.session_state.fmcg_dataset = map_season_month(st.session_state.fmcg_dataset)
            st.session_state.fmcg_dataset["is_public_holiday"] = st.session_state.fmcg_dataset[
                ["month", "day_of_the_month"]].apply(
                map_public_holiday, axis=1, args=(public_holidays_list,))
        else:
            st.warning("Download the dataset first.")

    st.dataframe(
        st.session_state.fmcg_dataset if st.session_state.fmcg_dataset is None else st.session_state.fmcg_dataset.head())

    st.write("---")
    st.write("## New Target Data")

    if "target_data" not in st.session_state:
        st.session_state.target_data = None

    if st.button("Create New Target Data"):
        if st.session_state.fmcg_dataset is not None:
            # Create a new target data
            st.session_state.target_data = st.session_state.fmcg_dataset.copy()
            st.session_state.target_data = st.session_state.target_data.loc[(st.session_state.target_data["sku"] == "YO-009")
                                                           & (st.session_state.target_data["region"] == "PL-North")
                                                           & (st.session_state.target_data["channel"] == "Retail")]
            st.session_state.target_data = st.session_state.target_data.groupby(by=["date", "season"], as_index=False).agg(
                units_sold=("units_sold", "sum"), promotion_flag=("promotion_flag", "first"), price_unit=("price_unit", "first"),
                pack_type=("pack_type", "first"), delivery_days=("delivery_days", "first"), year=("year", "first"),
                month=("month", "first"), week=("week", "first"), day_of_the_month=("day_of_the_month", "first"),
                day_of_the_week=("day_of_the_week", "first"), is_weekend=("is_weekend", "first"),
                is_public_holiday=("is_public_holiday", "first"), sub_season=("sub_season", "first"))

            # Search for missing dates
            date_range = pd.date_range(start=st.session_state.target_data["date"].min(),
                                       end=st.session_state.target_data["date"].max(), freq='D')

            # Impute missing dates
            st.session_state.target_data = pd.concat([st.session_state.target_data,
                                     pd.DataFrame(date_range.difference(st.session_state.target_data["date"]),
                                                  columns=["date"])], ignore_index=True)

            st.session_state.target_data.sort_values(by=["date"], inplace=True)
            st.session_state.target_data.reset_index(drop=True, inplace=True)

            # Reapply feature engineering and impute NaN
            st.session_state.target_data = chunk_date(st.session_state.target_data)
            st.session_state.target_data = categorize_weekend(st.session_state.target_data)
            st.session_state.target_data = map_season_month(st.session_state.target_data)
            st.session_state.target_data["is_public_holiday"] = st.session_state.target_data[["month", "day_of_the_month"]].apply(
                map_public_holiday, axis=1, args=(public_holidays_list,))
            st.session_state.target_data["units_sold"] = st.session_state.target_data["units_sold"].fillna(0)
            st.session_state.target_data["units_sold"] = st.session_state.target_data["units_sold"].astype(int)
            st.session_state.target_data["promotion_flag"] = st.session_state.target_data["promotion_flag"].fillna(0).astype(int)
            st.session_state.target_data.drop(["pack_type"], axis=1, inplace=True)
            st.session_state.target_data["price_unit"] = st.session_state.target_data["price_unit"].fillna(st.session_state.target_data.groupby(
                ["year", "month"])["price_unit"].transform("median")).round(2)
            st.session_state.target_data["delivery_days"] = st.session_state.target_data["delivery_days"].fillna(st.session_state.target_data.groupby(
                ["year", "month"])["delivery_days"].transform("median")).round(2)
            st.session_state.target_data.sort_values(by=["date"], inplace=True)
            st.session_state.target_data.reset_index(drop=True, inplace=True)
            st.session_state.target_data.sort_values(by=["date"], inplace=True)
            st.session_state.target_data.reset_index(drop=True, inplace=True)
            st.session_state.target_data["lag_7"] = st.session_state.target_data["units_sold"].shift(7).fillna(0)
            st.session_state.target_data["lag_7"] = st.session_state.target_data["lag_7"].astype(int)
            st.session_state.target_data["moving_average_7"] = st.session_state.target_data["units_sold"].rolling(window=7).mean().fillna(0).round(2)
        else:
            st.warning("Download the dataset first.")

    st.dataframe(
        st.session_state.target_data if st.session_state.target_data is None else st.session_state.target_data.head())

    st.write("---")
    st.write("## Apply Model")
    model = st.radio(
        "Choose model",
        ["Naive", "Season-Block Naive", "SARIMA", "XGBoost"],
        horizontal=True
    )

    if model == "Naive":
        if st.session_state.target_data is not None:
            # Naive Forecast
            naive_data = st.session_state.target_data.copy()
            naive_data_train = naive_data[naive_data["date"] < "2024-07-01"].copy()
            naive_data_test = naive_data[naive_data["date"] >= "2024-07-01"].copy()
            naive_data_test["naive_prediction"] = naive_data_test["units_sold"].shift(1).fillna(
                naive_data_train["units_sold"].iloc[-1]).astype(int)
            naive_data_test["naive_prediction_error"] = naive_data_test["units_sold"] - naive_data_test["naive_prediction"]
            naive_data_test["naive_prediction_absolute_error"] = naive_data_test["naive_prediction_error"].abs()
            st.session_state.mae = naive_data_test['naive_prediction_absolute_error'].mean().round(2)

            fig, ax = plt.subplots(figsize=(25, 10))
            ax.plot(naive_data_test["date"], naive_data_test["units_sold"], linewidth=0.6, label="Actual")
            ax.plot(naive_data_test["date"], naive_data_test["naive_prediction"], linewidth=0.6,
                    label="Prediction")
            ax.plot(st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                    st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                              st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                              st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                                              st.session_state.fmcg_dataset["stock_available"] == 0)][
                        "date"],
                    st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                            st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                              st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                              st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                                              st.session_state.fmcg_dataset["stock_available"] == 0)][
                        "units_sold"], "r.", markersize=9.0,
                    label="Stockout")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend(loc="upper left")
            ax.grid()
            plt.rcParams['font.family'] = 'garamond'
            plt.rcParams['font.size'] = 25
            plt.tight_layout()
            plt.xticks(rotation=30)
            st.pyplot(fig)

            st.write(f"MASE: {st.session_state.mae / st.session_state.mae}")
        else:
            st.warning("Create a new target data first.")
    elif model == "Season-Block Naive":
        if st.session_state.target_data is not None:
            # Season-Block Naive Forecast
            sb_naive_data = st.session_state.target_data.copy()
            sb_naive_data_train = sb_naive_data[sb_naive_data["date"] < "2024-07-01"].copy()
            sb_naive_data_test = sb_naive_data[sb_naive_data["date"] >= "2024-07-01"].copy()
            sb_naive_data_test["sb_naive_prediction"] = sb_naive_data_train[
                (sb_naive_data_train["date"] >= "2023-07-01") & (sb_naive_data_train["date"] <= "2023-12-31")][
                "units_sold"].values

            sb_naive_data_test["sb_error"] = sb_naive_data_test["units_sold"] - sb_naive_data_test["sb_naive_prediction"]
            sb_naive_data_test["sb_absolute_error"] = sb_naive_data_test["sb_error"].abs()
            sb_naive_data_test["sb_absolute_error/mae"] = sb_naive_data_test["sb_absolute_error"] / st.session_state.mae
            sb_naive_mase = sb_naive_data_test['sb_absolute_error/mae'].sum() / len(sb_naive_data_test)

            fig, ax = plt.subplots(figsize=(25, 10))
            ax.plot(sb_naive_data_test["date"], sb_naive_data_test["units_sold"], linewidth=0.6, label="Actual")
            ax.plot(sb_naive_data_test["date"], sb_naive_data_test["sb_naive_prediction"], linewidth=0.6,
                    label="Prediction")
            ax.plot(st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                        st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                              st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                  st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                                              st.session_state.fmcg_dataset["stock_available"] == 0)][
                        "date"],
                    st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                                st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                              st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                  st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                                              st.session_state.fmcg_dataset["stock_available"] == 0)][
                        "units_sold"], "r.", markersize=9.0,
                    label="Stockout")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend(loc="upper left")
            ax.grid()
            plt.rcParams['font.family'] = 'garamond'
            plt.rcParams['font.size'] = 25
            plt.tight_layout()
            plt.xticks(rotation=30)
            st.pyplot(fig)

            st.write(f"MASE: {sb_naive_mase}")
        else:
            st.warning("Create a new target data first.")
    elif model == "SARIMA":
        if st.session_state.target_data is not None:
            # SARIMA
            grid_search = st.radio(
                "Toggle Grid Search",
                options=["Default", "Grid Search"],
            )

            if grid_search == "Grid Search":
                st.warning("Original grid search takes ~15 minutes to finish. This is a shortcut version.")
                sarima_data = st.session_state.target_data.copy()

                # order_combinations = list(product([0, 1, 2], repeat=3))
                # seasonal_order_combinations = [x + (12,) for x in product([0, 1, 2], repeat=3)]
                # mase_dict = sarimax_grid_search(sarima_data, order_combinations, seasonal_order_combinations, st.session_state.mae)
                # best_sarima_model = mase_dict[min(mase_dict.keys())]

                sarima_data_train = sarima_data[(sarima_data["date"] < "2024-07-01")].copy()
                sarima_data_test = sarima_data[(sarima_data["date"] >= "2024-07-01")].copy()
                # sarima_regressor = SARIMAX(sarima_data_train["units_sold"], order=best_sarima_model[:3], seasonal_order=best_sarima_model[3:])
                sarima_regressor = SARIMAX(sarima_data_train["units_sold"], order=(1, 2, 2), seasonal_order=(0, 2, 2, 12))
                sarima_data_test["prediction"] = sarima_regressor.fit().forecast(steps=184).round(2).values
                sarima_data_test["sarima_error"] = sarima_data_test["units_sold"] - sarima_data_test["prediction"]
                sarima_data_test["sarima_absolute_error"] = sarima_data_test["sarima_error"].abs()
                sarima_data_test["sarima_absolute_error/mae"] = sarima_data_test[
                                                                    "sarima_absolute_error"] / st.session_state.mae
                sarima_mase = sarima_data_test['sarima_absolute_error/mae'].sum() / len(sarima_data_test)

                fig, ax = plt.subplots(figsize=(25, 10))
                ax.plot(sarima_data_test["date"], sarima_data_test["units_sold"], linewidth=.6, label="Actual")
                ax.plot(sarima_data_test["date"], sarima_data_test["prediction"], linewidth=.6, label="Prediction")
                ax.plot(st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                        st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                                  st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                  st.session_state.fmcg_dataset[
                                                                      "channel"] == "Retail") & (
                                                                  st.session_state.fmcg_dataset["stock_available"] == 0)][
                            "date"],
                        st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                                st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                                  st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                  st.session_state.fmcg_dataset[
                                                                      "channel"] == "Retail") & (
                                                                  st.session_state.fmcg_dataset["stock_available"] == 0)][
                            "units_sold"], "r.",
                        markersize=8.0, label="Stockout")
                ax.set_xlabel("date")
                ax.set_ylabel("units_sold")
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
                ax.legend()
                ax.grid(True)
                plt.rcParams['font.family'] = 'garamond'
                plt.rcParams['font.size'] = 25
                plt.tight_layout()
                plt.xticks(rotation=30)
                st.pyplot(fig)

                st.write(f"MASE: {sarima_mase}")
            else:
                sarima_data = st.session_state.target_data.copy()
                sarima_data_train = sarima_data[(sarima_data["date"] < "2024-07-01")].copy()
                sarima_data_test = sarima_data[(sarima_data["date"] >= "2024-07-01")].copy()
                sarima_regressor = SARIMAX(sarima_data_train["units_sold"], order=(2, 0, 2), seasonal_order=(0, 0, 0, 12))
                sarima_data_test["prediction"] = sarima_regressor.fit().forecast(steps=184).round(2).values
                sarima_data_test["sarima_error"] = sarima_data_test["units_sold"] - sarima_data_test["prediction"]
                sarima_data_test["sarima_absolute_error"] = sarima_data_test["sarima_error"].abs()
                sarima_data_test["sarima_absolute_error/mae"] = sarima_data_test[
                                                                    "sarima_absolute_error"] / st.session_state.mae
                sarima_mase = sarima_data_test['sarima_absolute_error/mae'].sum() / len(sarima_data_test)

                fig, ax = plt.subplots(figsize=(25, 10))
                ax.plot(sarima_data_test["date"], sarima_data_test["units_sold"], linewidth=.6, label="Actual")
                ax.plot(sarima_data_test["date"], sarima_data_test["prediction"], linewidth=.6, label="Prediction")
                ax.plot(st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                            st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                                  st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                      st.session_state.fmcg_dataset[
                                                                          "channel"] == "Retail") & (
                                                                  st.session_state.fmcg_dataset["stock_available"] == 0)][
                            "date"],
                        st.session_state.fmcg_dataset.loc[(st.session_state.fmcg_dataset["date"] >= "2024-06-01") & (
                                    st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                                                  st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                                                      st.session_state.fmcg_dataset[
                                                                          "channel"] == "Retail") & (
                                                                  st.session_state.fmcg_dataset["stock_available"] == 0)][
                            "units_sold"], "r.",
                        markersize=8.0, label="Stockout")
                ax.set_xlabel("date")
                ax.set_ylabel("units_sold")
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
                ax.legend()
                ax.grid(True)
                plt.rcParams['font.family'] = 'garamond'
                plt.rcParams['font.size'] = 25
                plt.tight_layout()
                plt.xticks(rotation=30)
                st.pyplot(fig)

                st.write(f"MASE: {sarima_mase}")
        else:
            st.warning("Create a new target data first.")
    else:
        if st.session_state.target_data is not None:
            # XGBoost
            xgboost_data = st.session_state.target_data.copy()
            xgboost_data["month"] = xgboost_data["month"].astype("category")
            xgboost_data["day_of_the_week"] = xgboost_data["day_of_the_week"].astype("category")
            xgboost_data["is_weekend"] = xgboost_data["is_weekend"].astype("category")
            xgboost_data["is_public_holiday"] = xgboost_data["is_public_holiday"].astype("category")
            xgboost_data["promotion_flag"] = xgboost_data["promotion_flag"].astype("category")
            xgboost_data["season"] = xgboost_data["season"].astype("category")
            xgboost_data["sub_season"] = xgboost_data["sub_season"].astype("category")
            xgboost_data.reset_index(drop=True, inplace=True)
            xgboost_data_train = xgboost_data[xgboost_data["date"] < "2024-07-01"].copy()
            xgboost_data_test = xgboost_data[xgboost_data["date"] >= "2024-07-01"].copy()

            FEATURES = ["month", "week", "day_of_the_month", "day_of_the_week", "season", "sub_season", "is_weekend",
                        "is_public_holiday", "price_unit", "promotion_flag", "delivery_days", "lag_7", "moving_average_7"]
            TARGET = "units_sold"
            X_train = xgboost_data_train[FEATURES]
            y_train = xgboost_data_train[TARGET]
            X_test = xgboost_data_test[FEATURES]
            y_test = xgboost_data_test[TARGET]
            regressor = xgb.XGBRegressor(base_score=.5, booster="gbtree", n_estimators=1000, early_stopping_rounds=100,
                                         objective="reg:squarederror", max_depth=100, learning_rate=0.01,
                                         enable_categorical=True)

            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
            xgboost_data_test["prediction"] = regressor.predict(X_test)
            xgboost_data_test["prediction"] = xgboost_data_test["prediction"].round(2)

            fig, ax = plt.subplots(figsize=(25, 10))
            ax.plot(xgboost_data_test["date"], xgboost_data_test["units_sold"], linewidth=.6, label="Actual")
            ax.plot(xgboost_data_test["date"], xgboost_data_test["prediction"], linewidth=1.0, label="Prediction")
            ax.plot(st.session_state.fmcg_dataset.loc[
                        (st.session_state.fmcg_dataset["date"] >= "2024-07-01") & (
                                    st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                    st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                st.session_state.fmcg_dataset["stock_available"] == 0)]["date"],
                    st.session_state.fmcg_dataset.loc[
                        (st.session_state.fmcg_dataset["date"] >= "2024-07-01") & (
                                    st.session_state.fmcg_dataset["region"] == "PL-North") & (
                                st.session_state.fmcg_dataset["sku"] == "YO-009") & (
                                    st.session_state.fmcg_dataset["channel"] == "Retail") & (
                                st.session_state.fmcg_dataset["stock_available"] == 0)]["units_sold"], "r.", markersize=8.0,
                    label="Stockout")
            ax.set_xlabel("date")
            ax.set_ylabel("units_sold")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.legend()
            ax.grid(True)
            plt.rcParams['font.family'] = 'garamond'
            plt.rcParams['font.size'] = 25
            plt.tight_layout()
            plt.xticks(rotation=30)
            st.pyplot(fig)

            xgboost_data_test["error"] = xgboost_data_test["units_sold"] - xgboost_data_test["prediction"]
            xgboost_data_test["absolute_error"] = xgboost_data_test["error"].abs()
            xgboost_data_test["absolute_error/mae"] = xgboost_data_test["absolute_error"]/st.session_state.mae
            xgboost_mase = xgboost_data_test['absolute_error/mae'].sum()/len(xgboost_data_test)

            st.write(f"MASE: {xgboost_mase}")
        else:
            st.warning("Create a new target data first.")

if __name__ == "__main__":
    main()