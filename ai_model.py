import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split  # will be used for data split
from sklearn.preprocessing import LabelEncoder  # for preprocessing
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)  # Scaling continious values as sd=1
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)  # for training the algorithm
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)  # for training the algorithm
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def scale_continious_features(data: pd.DataFrame, continious_features: list):
    scaler = StandardScaler()
    data.loc[:, continious_features] = scaler.fit_transform(data[continious_features])
    return data


def linear_regression(file: str, model_name: str):
    # Load the data from CSV
    data = pd.read_csv(file)

    # Define features and target
    continuous_features = [
        "room",
        "living_room",
        "net_sqm",
        "gross_sqm",
        "age",
        "latitude",
        "longitude",
    ]
    categorical_features = [
        "floor",
        "total_floor",
        "bathroom",
        "heating",
        "fuel",
        "usage",
        "credit",
        "furnished",
        "realty_type",
        "district",
        "county",
        "city",
    ]

    # Drop columns with more than 90% missing values
    data = data.dropna(thresh=len(data) * 0.1, axis=1)
    boolean_features = [col for col in data.columns if "_attributes" in col]

    for col in boolean_features:
        data[col] = data[col].fillna(0)
        data[col] = data[col].astype(bool)

    categorical_features.extend(boolean_features)

    target_feature = "price"

    X = data.drop(columns=[target_feature])
    y = data[target_feature]
    X["updated_at"] = X["updated_at"].astype(str).str[:7]
    unique_months = X["updated_at"].unique().tolist()
    continues_values_for_updated_month = [i + 1 for i, _ in enumerate(unique_months)]
    continues_values_for_updated_month.reverse()
    X["updated_at"] = X["updated_at"].replace(
        unique_months, continues_values_for_updated_month
    )
    continuous_features.append("updated_at")
    X = scale_continious_features(X, continuous_features)

    # Preprocessing pipeline
    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Handling missing values and encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Create linear regression model pipeline
    lr_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    # Train the model
    lr_pipeline.fit(X_train, y_train)

    # Evaluate the model
    train_predictions = lr_pipeline.predict(X_train)
    val_predictions = lr_pipeline.predict(X_val)

    joblib.dump(lr_pipeline, model_name)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Train RMSE: {train_rmse}")
    print(f"Validation RMSE: {val_rmse}")


class BaseRandomForestRegressor(ABC):
    @abstractmethod
    def __init__(self):
        self.suffix = ""
        self.path_to_artifacts = "bin"
        self.input_data = pd.DataFrame()
        self.reverse_input_data = pd.DataFrame()
        self._target_column = "price"
        self._target = None
        self.unnecessary_columns = [
            "id",
            "internal_id",
            "data_source",
            "url",
            "version",
            "is_last_version",
            "created_at",
            "inserted_at",
            "predicted_price",
            "predicted_rental_price",
            "is_active",
            "listing_category",
            "county",
            "district",
            "credit",
            "deposit",
            "price",
        ]

    @property
    def input_data(self):
        return self._input_data

    @input_data.setter
    def input_data(self, value):
        self._input_data = value
        self._update_target()

    @property
    def target_column(self):
        return self._target_column

    @target_column.setter
    def target_column(self, value):
        self._target_column = value
        self._update_target()

    @property
    def target(self):
        return self._target

    def _update_target(self):
        if (
            not self._input_data.empty
            and self._target_column in self._input_data.columns
        ):
            self._target = self._input_data[self._target_column]

    def remove_highly_constant_columns(self, dataframe, threshold=0.9):
        # Calculate the threshold for 90% of the rows
        row_count = len(dataframe)
        threshold_count = row_count * threshold

        # Identify columns to drop
        columns_to_drop = []
        non_drop_columns = ["realty_type", "location_attributes_manzara___deniz"]
        for column in dataframe.columns:
            if column in non_drop_columns:
                continue

            # Get the most frequent value count
            most_frequent_value_count = dataframe[column].value_counts().max()
            # Check if the most frequent value count exceeds the threshold
            if most_frequent_value_count > threshold_count:
                columns_to_drop.append(column)

        # Drop the identified columns
        dataframe = dataframe.drop(columns=columns_to_drop)
        return dataframe

    def convert_date_to_integer(
        self, dataframe: pd.DataFrame, date_field: str = "updated_at"
    ):
        # Converts year-month format into a unique integer
        dataframe.loc[:, date_field] = dataframe[date_field].astype(str).str[:7]
        unique_months = dataframe[date_field].unique().tolist()
        continues_values_for_inserted_month = [
            i + 1 for i, _ in enumerate(unique_months)
        ]
        continues_values_for_inserted_month.reverse()
        dataframe.loc[:, date_field] = dataframe[date_field].replace(
            unique_months, continues_values_for_inserted_month
        )
        date_to_int_map = {
            date: index
            for date, index in zip(unique_months, continues_values_for_inserted_month)
        }

        joblib.dump(
            date_to_int_map,
            f"./{self.path_to_artifacts}/date_index_map_{self.suffix}.joblib",
            compress=True,
        )

        return date_to_int_map

    @abstractmethod
    def preprocess(self):
        """Preprocess the input data"""

    def fill_nones_with_zeros(self, dataframe: pd.DataFrame, continious_features: list):
        # Fill None values continious_features with 0.
        df_slice = dataframe[self.continious_features].copy()
        df_slice.fillna(0, inplace=True)
        dataframe.loc[:, continious_features] = df_slice

    def convert_categoricals(self, dataframe: pd.DataFrame):
        encoders = {}
        for column in self.categorical_features:
            categorical_convert = LabelEncoder()
            dataframe.loc[:, column] = categorical_convert.fit_transform(
                dataframe[column]
            )
            encoders[column] = categorical_convert

        return encoders

    @abstractmethod
    def postprocess(self):
        """Postprocess after training the model"""

    def feature_importances(
        self, model: RandomForestRegressor, X_train: pd.DataFrame
    ) -> dict:
        importances = model.feature_importances_
        features = X_train.columns
        return {
            feature: importance for feature, importance in zip(features, importances)
        }


class MyRandomForestRegressorForSatilikHouses(BaseRandomForestRegressor):
    def __init__(self):
        super().__init__()
        self.suffix = "satilik"
        self.input_data = pd.read_csv(f"exports/2024-10-05-sanitized-satilik.csv")
        self.reverse_input_data = pd.read_csv(
            f"exports/2024-10-05-sanitized-kiralik.csv"
        )
        self.input_data.dropna(thresh=len(self.input_data) * 0.1, axis=1, inplace=True)
        self.reverse_input_data.dropna(
            thresh=len(self.reverse_input_data) * 0.1, axis=1, inplace=True
        )
        self.input_data = self.remove_highly_constant_columns(self.input_data)
        self.reverse_input_data = self.remove_highly_constant_columns(
            self.reverse_input_data
        )
        self.input_data["county_district"] = (
            self.input_data["county"] + "-" + self.input_data["district"]
        )
        self.reverse_input_data["county_district"] = (
            self.reverse_input_data["county"]
            + "-"
            + self.reverse_input_data["district"]
        )
        self.all_columns = [
            column
            for column in self.input_data.columns.to_list()
            if column not in self.unnecessary_columns and "_attributes" not in column
        ]
        self.all_columns_reverse = [
            column
            for column in self.reverse_input_data.columns.to_list()
            if column not in self.unnecessary_columns and "_attributes" not in column
        ]
        self.common_columns = [
            c for c in set(self.all_columns).intersection(set(self.all_columns_reverse))
        ]

        # Only get the columns are in self.all_columns
        self.continious_features = [
            col
            for col in self.all_columns
            if col
            in [
                "room",
                "living_room",
                "bathroom",
                "age",
                "latitude",
                "longitude",
                "net_sqm",
                "gross_sqm",
                "deposit",
                "updated_at",
            ]
        ]
        self.categorical_features = [
            col
            for col in self.all_columns
            if col
            not in (self.continious_features + [self.target_column, "updated_at"])
            and "_attributes" not in col
        ]

    @property
    def _clean_dataframe(self):
        return self.input_data[
            self.continious_features + self.categorical_features + [self.target_column]
        ]

    def preprocess(self):
        # self._convert_booleans(self.clean_dataframe, self.boolean_features)
        self.preprocessed_dataframe = self._clean_dataframe
        encoders = self.convert_categoricals(self.preprocessed_dataframe)
        date_index_map = self.convert_date_to_integer(self.preprocessed_dataframe)
        self.fill_nones_with_zeros(
            self.preprocessed_dataframe, self.continious_features
        )
        scale_continious_features(self.preprocessed_dataframe, self.continious_features)

        joblib.dump(encoders, f"./bin/encoders_{self.suffix}.joblib", compress=True)
        joblib.dump(
            date_index_map, f"./bin/date_index_map_{self.suffix}.joblib", compress=True
        )

    def postprocess(self, model: RandomForestRegressor, X_train: pd.DataFrame):
        # save preprocessing objects and RF algorithm
        joblib.dump(
            self.feature_importances(model, X_train),
            f"./bin/feature_importances_{self.suffix}.joblib",
            compress=True,
        )
        joblib.dump(
            model,
            f"./bin/random_forest_without_scaling_{self.suffix}.joblib",
            compress=True,
        )
        # joblib.dump(et, "./bin/extra_trees_without_scaling.joblib", compress=True)

    def train(self):
        self.preprocess()

        X = self.preprocessed_dataframe[
            self.continious_features + self.categorical_features
        ]
        y = self.preprocessed_dataframe[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        imputer = SimpleImputer(strategy="mean")
        pipeline = make_pipeline(imputer, rf)
        rf_result = pipeline.fit(X_train, y_train)

        self.postprocess(rf, X_train)

        # Predictions with test data
        y_predict_with_rf = rf_result.predict(X_test)
        print(r2_score(y_test.values, y_predict_with_rf))
        print(mean_squared_error(y_test.values, y_predict_with_rf))

    @staticmethod
    def _convert_booleans(data: pd.DataFrame, boolean_features: list):
        for col in boolean_features:
            data[col] = data[col].fillna(0)
            data[col] = data[col].astype(bool)

        return data

    def _encode_categorical_features(self, dataframe: pd.DataFrame):
        # encode categorical features
        for column in self.all_columns:
            # scanning in categorical and continious features
            try:
                # If the feature is categorical then encode the categories
                encoders = joblib.load(
                    f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib"
                )
                categorical_convert = encoders[column]
                dataframe[column] = categorical_convert.transform(dataframe[column])
            except:
                # If the feature is not categorical the add as it is
                logger.info(f"Column {column} is not categorical")


class MyRandomForestRegressorForRentalHouses(BaseRandomForestRegressor):
    def __init__(self):
        super().__init__()
        self.suffix = "kiralik"
        self.input_data = pd.read_csv(f"exports/2024-10-05-sanitized-kiralik.csv")
        self.reverse_input_data = pd.read_csv(
            f"exports/2024-10-05-sanitized-satilik.csv"
        )
        self.input_data.dropna(thresh=len(self.input_data) * 0.1, axis=1, inplace=True)
        self.reverse_input_data.dropna(
            thresh=len(self.reverse_input_data) * 0.1, axis=1, inplace=True
        )
        self.input_data = self.remove_highly_constant_columns(self.input_data)
        self.reverse_input_data = self.remove_highly_constant_columns(
            self.reverse_input_data
        )
        self.input_data["county_district"] = (
            self.input_data["county"] + "-" + self.input_data["district"]
        )
        self.reverse_input_data["county_district"] = (
            self.reverse_input_data["county"]
            + "-"
            + self.reverse_input_data["district"]
        )
        self.all_columns = [
            column
            for column in self.input_data.columns.to_list()
            if column not in self.unnecessary_columns and "_attributes" not in column
        ]
        self.all_columns_reverse = [
            column
            for column in self.reverse_input_data.columns.to_list()
            if column not in self.unnecessary_columns and "_attributes" not in column
        ]
        self.common_columns = [
            c for c in set(self.all_columns).intersection(set(self.all_columns_reverse))
        ]

        # Only get the columns are in self.all_columns
        self.continious_features = [
            col
            for col in self.all_columns
            if col
            in [
                "room",
                "living_room",
                "bathroom",
                "age",
                "latitude",
                "longitude",
                "net_sqm",
                "gross_sqm",
                "deposit",
                "updated_at",
            ]
        ]
        self.categorical_features = [
            col
            for col in self.all_columns
            if col
            not in (self.continious_features + [self.target_column, "updated_at"])
            and "_attributes" not in col
        ]

    @property
    def _clean_dataframe(self):
        return self.input_data[
            self.continious_features + self.categorical_features + [self.target_column]
        ]

    def preprocess(self):
        # self._convert_booleans(self.clean_dataframe, self.boolean_features)
        self.preprocessed_dataframe = self._clean_dataframe
        encoders = self.convert_categoricals(self.preprocessed_dataframe)
        date_index_map = self.convert_date_to_integer(self.preprocessed_dataframe)
        self.fill_nones_with_zeros(
            self.preprocessed_dataframe, self.continious_features
        )
        scale_continious_features(self.preprocessed_dataframe, self.continious_features)

        joblib.dump(encoders, f"./bin/encoders_{self.suffix}.joblib", compress=True)
        joblib.dump(
            date_index_map, f"./bin/date_index_map_{self.suffix}.joblib", compress=True
        )

    def postprocess(self, model: RandomForestRegressor, X_train: pd.DataFrame):
        # save preprocessing objects and RF algorithm
        joblib.dump(
            self.feature_importances(model, X_train),
            f"./bin/feature_importances_{self.suffix}.joblib",
            compress=True,
        )
        joblib.dump(
            model,
            f"./bin/random_forest_without_scaling_{self.suffix}.joblib",
            compress=True,
        )
        # joblib.dump(et, "./bin/extra_trees_without_scaling.joblib", compress=True)

    def train(self):
        self.preprocess()

        X = self.preprocessed_dataframe[
            self.continious_features + self.categorical_features
        ]
        y = self.preprocessed_dataframe[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        imputer = SimpleImputer(strategy="mean")
        pipeline = make_pipeline(imputer, rf)
        rf_result = pipeline.fit(X_train, y_train)

        self.postprocess(rf, X_train)

        # Predictions with test data
        y_predict_with_rf = rf_result.predict(X_test)
        print(r2_score(y_test.values, y_predict_with_rf))
        print(mean_squared_error(y_test.values, y_predict_with_rf))

    @staticmethod
    def _convert_booleans(data: pd.DataFrame, boolean_features: list):
        for col in boolean_features:
            data[col] = data[col].fillna(0)
            data[col] = data[col].astype(bool)

        return data

    def _encode_categorical_features(self, dataframe: pd.DataFrame):
        # encode categorical features
        for column in self.all_columns:
            # scanning in categorical and continious features
            try:
                # If the feature is categorical then encode the categories
                encoders = joblib.load(
                    f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib"
                )
                categorical_convert = encoders[column]
                dataframe[column] = categorical_convert.transform(dataframe[column])
            except:
                # If the feature is not categorical the add as it is
                logger.info(f"Column {column} is not categorical")


class MyRandomForestPredictor:
    def __init__(self, input_data, suffix="satilik") -> None:
        self.suffix = suffix
        self.path_to_artifacts = "bin"
        self.input_data = input_data
        self.input_data["county_district"] = (
            self.input_data["county"] + "-" + self.input_data["district"]
        )
        self.all_columns = input_data.columns.to_list()
        self.identifiers = self.input_data[["internal_id", "data_source", "price"]]
        self.unnecessary_columns = [
            col
            for col in [
                "id",
                "internal_id",
                "data_source",
                "url",
                "version",
                "is_last_version",
                "created_at",
                "inserted_at",
                "predicted_price",
                "predicted_rental_price",
                "is_active",
                "listing_category",
                "district",
                "county",
            ]
            if col in self.all_columns
        ]
        self.continious_features = [
            col
            for col in [
                "room",
                "living_room",
                "age",
                "latitude",
                "longitude",
                "net_sqm",
                "gross_sqm",
                "deposit",
            ]
            if col in self.all_columns
        ]
        self.categorical_features = [
            col
            for col in self.all_columns
            if col
            not in (
                self.continious_features
                + self.unnecessary_columns
                + ["price", "updated_at", "listing_category"]
            )
            and "_attributes" not in col
        ]
        self.data = self.input_data[
            self.continious_features + self.categorical_features + ["updated_at"]
        ]

    @staticmethod
    def _convert_booleans(data: pd.DataFrame, boolean_features: list):
        for col in boolean_features:
            data[col] = data[col].fillna(0)
            data[col] = data[col].astype(bool)

        return data

    def _encode_categorical_features(self, dataframe: pd.DataFrame):
        # encode categorical features
        encoders = joblib.load(
            f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib"
        )
        for column in self.data.columns.to_list():
            # scanning in categorical and continious features
            try:
                # If the feature is categorical then encode the categories
                categorical_convert = encoders[column]
                dataframe.loc[:, column] = categorical_convert.transform(
                    dataframe[column]
                )
            except:
                # If the feature is not categorical the add as it is
                dataframe.loc[:, column] = dataframe[column]

    def _convert_datetime_to_continious_integer(
        self, dataframe: pd.DataFrame, date_field: str
    ):
        # Converts year-month format into a unique integer
        converter = joblib.load(
            f"./{self.path_to_artifacts}/date_index_map_{self.suffix}.joblib"
        )
        dataframe.loc[:, date_field] = dataframe[date_field].astype(str).str[:7]
        dataframe.loc[:, date_field] = dataframe[date_field].replace(
            converter.keys(), converter.values()
        )

    def predict(self):
        self.values_fill_missing = joblib.load(
            f"./{self.path_to_artifacts}/most_frequent_values_{self.suffix}.joblib"
        )
        self.encoders = joblib.load(
            f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib"
        )
        model = joblib.load(
            f"./{self.path_to_artifacts}/random_forest_without_scaling_{self.suffix}.joblib"
        )
        feature_importances = joblib.load(
            f"./{self.path_to_artifacts}/feature_importances_{self.suffix}.joblib"
        )
        features = feature_importances.keys()
        data = self.input_data[features]
        self._encode_categorical_features(data)
        scale_continious_features(data, self.continious_features)
        self._convert_datetime_to_continious_integer(data, "updated_at")
        predictions = model.predict(data)
        print(predictions, self.identifiers)
        return predictions


def get_non_common_features(set1, set2):
    return list(set1.symmetric_difference(set2))


if __name__ == "__main__":
    regressor_satilik = MyRandomForestRegressorForSatilikHouses()
    regressor_kiralik = MyRandomForestRegressorForRentalHouses()

    regressor_satilik.train()
    regressor_kiralik.train()
    # linear_regression("exports/sorted-result-satilik.csv", "bin/linear_regression_model_satilik.pkl")
    # linear_regression("exports/sorted-result-kiralik.csv", "bin/linear_regression_model_kiralik.pkl")


## Playground
# df = pd.read_csv("exports/2024-10-05-sanitized-satilik.csv")
# predictor = MyRandomForestPredictor(input_data=pd.DataFrame(df.head().to_dict(orient="records")))
