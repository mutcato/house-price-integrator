import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split # will be used for data split
from sklearn.preprocessing import LabelEncoder # for preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Scaling continious values as sd=1
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # for training the algorithm
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor # for training the algorithm
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import numpy as np
import joblib


def scale_continious_features(data: pd.DataFrame, continious_features: list):
    scaler = StandardScaler()
    data[continious_features] = scaler.fit_transform(data[continious_features])
    return data

def linear_regression(file:str, model_name:str):
    # Load the data from CSV
    data = pd.read_csv(file)

    # Define features and target
    continuous_features = ['room', 'living_room', 'net_sqm', 'gross_sqm', 'age', 'latitude', 'longitude']
    categorical_features = ['floor', 'total_floor', 'bathroom', 'heating', 'fuel', 'usage', 'credit', 'furnished', 'realty_type', 'district', 'county', 'city']
    
    # Drop columns with more than 90% missing values
    data = data.dropna(thresh=len(data)*0.1, axis=1)   
    boolean_features = [col for col in data.columns if "_attributes" in col]

    for col in boolean_features:
        data[col] = data[col].fillna(0)
        data[col] = data[col].astype(bool)

    categorical_features.extend(boolean_features)

    target_feature = 'price'

    X = data.drop(columns=[target_feature])
    y = data[target_feature]
    X["updated_at"] = X["updated_at"].astype(str).str[:7]
    unique_months = X["updated_at"].unique().tolist()
    continues_values_for_updated_month = [i+1 for i, _ in enumerate(unique_months)]
    continues_values_for_updated_month.reverse()
    X["updated_at"] = X["updated_at"].replace(unique_months, continues_values_for_updated_month)
    continuous_features.append("updated_at")
    X = scale_continious_features(X, continuous_features)

    # Preprocessing pipeline
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # Handling missing values and encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features)])

    # X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create linear regression model pipeline
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])

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


class MyRandomForestRegressor:
    
    # Change instance variables to class variables
    
    def __init__(self, input_data: pd.DataFrame, suffix:str=""):
        # JSON to pandas DataFrame
        self.suffix = suffix
        self.path_to_artifacts = "bin"
        self.input_data = input_data
        self.input_data.dropna(thresh=len(self.input_data)*0.1, axis=1, inplace=True)
        self.target_column = 'price'
        self.target = self.input_data[self.target_column]
        self.all_columns = self.input_data.columns.to_list()
        
        # Only get the columns are in self.all_columns
        self.unnecessary_columns = [col for col in ['id', 'internal_id', 'data_source', 'url', 'version', 'is_last_version', 'created_at' ,'inserted_at', 'predicted_price', 'predicted_rental_price', 'is_active', 'listing_category'] if col in self.all_columns]
        
        # Only get the columns are in self.all_columns
        self.continious_features = [col for col in ['room', 'living_room', 'age', 'latitude', 'longitude', 'net_sqm', 'gross_sqm', 'deposit'] if col in self.all_columns]
        
        self.categorical_features = [col for col in self.all_columns if col not in (self.continious_features + [self.target_column, "updated_at"] + self.unnecessary_columns)]
        self.boolean_features = [col for col in self.input_data.columns if "_attributes" in col]

    def train(self):
        self.input_data = self._convert_booleans(self.input_data, self.boolean_features)
        self._convert_categoricals()
        date_index_map = self._convert_date_to_integer(self.input_data, "updated_at")
        
        # Drop columns with more than 90% missing values
        self.input_data = self.input_data.dropna(thresh=len(self.input_data)*0.1, axis=1)

        # Fill None values continious_features with 0. 
        df_slice = self.input_data[self.continious_features].copy()
        df_slice.fillna(0, inplace=True)
        self.input_data[self.continious_features] = df_slice

        scale_continious_features(self.input_data, self.continious_features)

        X = self.input_data[self.continious_features + self.categorical_features + self.boolean_features + ["updated_at"]]
        y = self.input_data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)
        rf = RandomForestRegressor(n_estimators = 100, n_jobs=-1)
        imputer = SimpleImputer(strategy='mean')
        pipeline = make_pipeline(imputer, rf)
        rf_result = pipeline.fit(X_train, y_train)

        # Predictions with test data
        y_predict_with_rf = rf_result.predict(X_test)
        print(r2_score(y_test.values, y_predict_with_rf))
        print(mean_squared_error(y_test.values, y_predict_with_rf))
        most_frequent_values = dict(self.input_data.mode().iloc[0])
        # save preprocessing objects and RF algorithm
        joblib.dump(most_frequent_values, f"./bin/most_frequent_values_{self.suffix}.joblib", compress=True)
        joblib.dump(self.encoders, f"./bin/encoders_{self.suffix}.joblib", compress=True)
        joblib.dump(rf, f"./bin/random_forest_without_scaling_{self.suffix}.joblib", compress=True)
        joblib.dump(date_index_map, f"./bin/date_index_map_{self.suffix}.joblib", compress=True)
        joblib.dump(self.boolean_features, f"./bin/attributes_{self.suffix}.joblib", compress=True)
        # joblib.dump(et, "./bin/extra_trees_without_scaling.joblib", compress=True)

    def _convert_date_to_integer(self, dataframe: pd.DataFrame, date_field: str):
        # Converts year-month format into a unique integer
        dataframe[date_field] = dataframe[date_field].astype(str).str[:7]
        unique_months = dataframe[date_field].unique().tolist()
        continues_values_for_inserted_month = [i+1 for i, _ in enumerate(unique_months)]
        continues_values_for_inserted_month.reverse()
        dataframe[date_field] = dataframe[date_field].replace(unique_months, continues_values_for_inserted_month)
        self.continious_features.append(date_field)
        return {date: index for date, index in zip(unique_months, continues_values_for_inserted_month)}

    @staticmethod
    def _convert_booleans(data: pd.DataFrame, boolean_features: list):
        for col in boolean_features:
            data[col] = data[col].fillna(0)
            data[col] = data[col].astype(bool)

        return data

    def _convert_categoricals(self):
        self.encoders = {}
        for column in self.categorical_features:
            categorical_convert = LabelEncoder()
            self.input_data[column] = categorical_convert.fit_transform(self.input_data[column])
            self.encoders[column] = categorical_convert    

    def preprocessing(self):
        # combine district and neighbourhood
        self.input_data['ilce_mah'] = self.input_data['ilce']+'-'+self.input_data['mah']

        # drop unneccassary columns
        self.processed_data = self.input_data[self.all_columns]
        
        # fill missing values
        self.processed_data.fillna(self.values_fill_missing, inplace=True)    
        
        # get categorical features
        self.categorical_features = [
            item for item in self.processed_data if item not in (self.continious_features + [self.target_column])
        ]
        
        # scale continious features [later phase]

        # encode categorical features
        encoded_data = {}
        for column in self.all_columns:
            # scanning in categorical and continious features
            try:
                # If the feature is categorical then encode the categories
                categorical_convert = self.encoders[column]
                encoded_data[column] = categorical_convert.transform(self.processed_data[column])
            except:   
                # If the feature is not categorical the add as it is
                encoded_data[column] = self.input_data[column]
        
        return encoded_data
    
    def _encode_categorical_features(self, dataframe: pd.DataFrame):
        # encode categorical features
        for column in self.all_columns:
            # scanning in categorical and continious features
            try:
                # If the feature is categorical then encode the categories
                encoders = joblib.load(f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib")
                categorical_convert = encoders[column]
                dataframe[column] = categorical_convert.transform(dataframe[column])
            except:   
                # If the feature is not categorical the add as it is
                dataframe[column] = dataframe[column]
        
    
    def _convert_datetime_to_continious_integer(self, dataframe: pd.DataFrame, date_field: str):
        # Converts year-month format into a unique integer
        converter = joblib.load(f"./{self.path_to_artifacts}/date_index_map_{self.suffix}.joblib")
        dataframe[date_field] = dataframe[date_field].astype(str).str[:7]
        dataframe[date_field] = dataframe[date_field].replace(converter.keys(), converter.values())
    


class MyRandomForestPredictor:
    def __init__(self, input_data, suffix="satilik") -> None:
        self.suffix = suffix
        self.path_to_artifacts = "bin"
        self.input_data = input_data
        self.all_columns = input_data.columns.to_list()
        self.continious_features = [col for col in ['room', 'living_room', 'age', 'latitude', 'longitude', 'net_sqm', 'gross_sqm', 'deposit'] if col in self.all_columns]
        self.categorical_features = [col for col in self.all_columns if col not in (self.continious_features + ["price", "updated_at", "listing_category"])]
        self.boolean_features = [col for col in input_data.columns if "_attributes" in col]
        self.data = input_data[self.continious_features + self.categorical_features + self.boolean_features + ["updated_at"]]

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
                encoders = joblib.load(f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib")
                categorical_convert = encoders[column]
                dataframe[column] = categorical_convert.transform(dataframe[column])
            except:   
                # If the feature is not categorical the add as it is
                dataframe[column] = dataframe[column]

    def _convert_datetime_to_continious_integer(self, dataframe: pd.DataFrame, date_field: str):
        # Converts year-month format into a unique integer
        converter = joblib.load(f"./{self.path_to_artifacts}/date_index_map_{self.suffix}.joblib")
        dataframe[date_field] = dataframe[date_field].astype(str).str[:7]
        dataframe[date_field] = dataframe[date_field].replace(converter.keys(), converter.values())

    def predict(self):
        self.values_fill_missing = joblib.load(f"./{self.path_to_artifacts}/most_frequent_values_{self.suffix}.joblib")
        self.encoders = joblib.load(f"./{self.path_to_artifacts}/encoders_{self.suffix}.joblib")
        model = joblib.load(f"./{self.path_to_artifacts}/random_forest_without_scaling_{self.suffix}.joblib")
        self.data1 = self._convert_booleans(self.input_data, self.boolean_features)
        self._encode_categorical_features(self.data1)
        scale_continious_features(self.data1, self.continious_features)
        self._convert_datetime_to_continious_integer(self.data1, "updated_at")
        breakpoint()
        return model.predict(self.data1)



def get_non_common_features(set1, set2):
    return list(set1.symmetric_difference(set2))

if __name__ == "__main__":
    satilik_csv_dataframe = pd.read_csv("exports/sorted-result-satilik.csv")
    regressor_satilik = MyRandomForestRegressor(satilik_csv_dataframe, suffix="satilik")
    kiralik_csv_dataframe = pd.read_csv("exports/sorted-result-kiralik.csv")
    regressor_kiralik = MyRandomForestRegressor(kiralik_csv_dataframe, suffix="kiralik")
    common_features = [c for c in set(regressor_satilik.input_data.columns).intersection(set(regressor_kiralik.input_data.columns))]
    non_common_features = get_non_common_features(
        set(regressor_satilik.input_data.columns), 
        set(regressor_kiralik.input_data.columns)
    )

    # Bu extend işlemi hiçbir boka yaramıyor. Regressor zaten init oldu
    regressor_satilik.unnecessary_columns.extend(non_common_features)
    regressor_kiralik.unnecessary_columns.extend(non_common_features)
    regressor_satilik.train()
    regressor_kiralik.train()
    # linear_regression("exports/sorted-result-satilik.csv", "bin/linear_regression_model_satilik.pkl")
    # linear_regression("exports/sorted-result-kiralik.csv", "bin/linear_regression_model_kiralik.pkl")