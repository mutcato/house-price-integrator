import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    X["inserted_at"] = X["inserted_at"].astype(str).str[:7]
    unique_months = X["inserted_at"].unique().tolist()
    continues_values_for_inserted_month = [i+1 for i, _ in enumerate(unique_months)]
    continues_values_for_inserted_month.reverse()
    X["inserted_at"] = X["inserted_at"].replace(unique_months, continues_values_for_inserted_month)
    continuous_features.append("inserted_at")
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

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

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


class MyRandomForestRegressor(object):
    
    # Change instance variables to class variables
    
    def __init__(self, file:str):
        # JSON to pandas DataFrame
        path_to_artifacts = "bin/"
        self.values_fill_missing = joblib.load(path_to_artifacts + "most_frequent_values.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest_without_scaling.joblib")

        self.input_data = pd.read_csv(file)
        self.input_data.dropna(thresh=len(self.input_data)*0.1, axis=1, inplace=True)
        self.target_column = 'price'
        self.target = self.input_data[self.target_column]
        self.ilan_no = self.input_data['ilan_no']
        self.all_columns = self.input_data.columns.to_list()
        self.unnecessary_columns = ['id', 'internal_id', 'data_source', 'url', 'version', 'is_last_version', 'created_at', 'updated_at', 'predicted_price', 'predicted_rental_price', 'is_active', 'listing_category']
        self.continious_features = ['room', 'living_room', 'age', 'latitude', 'longitude', 'net_sqm', 'gross_sqm']
        self.categorical_features = [col for col in self.all_columns if col not in (self.continious_features + [self.target_column] + self.unnecessary_columns)]

        
    def _convert_categoricals(self):
        encoders = {}
        for column in self.categorical_features:
            categorical_convert = LabelEncoder()
            self.input_data[column] = categorical_convert.fit_transform(self.input_data[column])
            encoders[column] = categorical_convert    

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

    def predict(self, input_data):
        input_df = pd.DataFrame.from_dict(input_data)
        return self.model.predict(input_df)

    def postprocessing(self, predictions):
        result = []
        for key, pred in enumerate(predictions):
            label = "ucuz"
            if pred < self.input_data[self.target_column][key]: label = "pahali"
                
            actual_price = self.input_data[self.target_column][key]
            result.append({
                "ilan_no": self.ilan_no[key],
                "actual": actual_price,
                "prediction": pred,
                "diff_nominal": round(actual_price - pred),
                "diff_percentage": round(100 * (actual_price - pred) / actual_price, 2),
                "label": label,
                "status": "OK"
            })
        
        return result

    def compute_prediction(self):
        try:
            encoded_data = self.preprocessing()
            
            prediction = self.predict(encoded_data)
            
            prediction = self.postprocessing(prediction)

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction


if __name__ == "__main__":
    # regressor = MyRandomForestRegressor("exports/sorted-result-satilik.csv")
    linear_regression("exports/sorted-result-satilik.csv", "bin/linear_regression_model_satilik.pkl")
    linear_regression("exports/sorted-result-kiralik.csv", "bin/linear_regression_model_kiralik.pkl")