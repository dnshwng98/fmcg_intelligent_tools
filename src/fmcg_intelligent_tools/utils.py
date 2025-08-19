import kagglehub
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from constants import SEASON_MONTH_PAIR, SUB_SEASON_MONTH_PAIR

def download_dataset(dataset_url: str) -> pd.DataFrame:
    path = kagglehub.dataset_download(dataset_url)
    fmcg_dataset = pd.read_csv(path + "\\FMCG_2022_2024.csv", parse_dates=["date"])
    fmcg_dataset.sort_values(by=["date", "category", "brand", "segment", "sku", "region", "channel"])
    fmcg_dataset.reset_index(drop=True, inplace=True)

    return fmcg_dataset

def create_month_day_series(date_dictionary: dict) -> list[str]:
    dates_list = []

    for date in date_dictionary.values():
        if isinstance(date, list):
            dates_list.extend(date)
        else:
            dates_list.append(date)

    return dates_list

def chunk_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month_name()
    dataframe["week"] = dataframe["date"].dt.isocalendar().week
    dataframe["day_of_the_month"] = dataframe["date"].dt.day
    dataframe["day_of_the_week"] = dataframe["date"].dt.day_name()

    return dataframe

def categorize_weekend(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["is_weekend"] = dataframe["day_of_the_week"].isin(["Saturday", "Sunday"])

    return dataframe

def map_season_month(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["season"] = dataframe["month"].map(SEASON_MONTH_PAIR)
    dataframe["sub_season"] = dataframe["month"].map(SUB_SEASON_MONTH_PAIR)

    return dataframe

def map_public_holiday(data, public_holidays_list: list) -> bool:
    return (data["month"] + " " + str(data["day_of_the_month"])) in (public_holidays_list)


def sarimax_grid_search(sarima_data: pd.DataFrame, order_combinations: list, seasonal_order_combinations: list,
                        mae: float) -> dict:
    mase_dict = {}

    for order_combination in order_combinations:
        for seasonal_order_combination in seasonal_order_combinations:
            try:
                sarima_data_train = sarima_data[(sarima_data["date"] < "2024-07-01")].copy()
                sarima_data_test = sarima_data[(sarima_data["date"] >= "2024-07-01")].copy()
                sarima_regressor = SARIMAX(sarima_data_train["units_sold"], order=order_combination,
                                           seasonal_order=seasonal_order_combination)
                sarima_data_test["prediction"] = sarima_regressor.fit().forecast(steps=184).round(2).values
                data_accuracy = sarima_data_test[["units_sold", "prediction"]].copy()
                data_accuracy["error"] = data_accuracy["units_sold"] - data_accuracy["prediction"]
                data_accuracy["absolute_error"] = data_accuracy["error"].abs()
                data_accuracy["absolute_error/mae"] = data_accuracy["absolute_error"] / mae
                mase_dict.update([(data_accuracy['absolute_error/mae'].sum() / len(data_accuracy),
                                   order_combination + seasonal_order_combination)])
            except np.linalg.LinAlgError:
                continue

    return mase_dict