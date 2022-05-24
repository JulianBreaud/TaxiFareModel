# imports
import pandas as pd
import mlflow

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from memoized_property import memoized_property
from  mlflow.tracking import MlflowClient

# Connexion a MLFLOW
MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[FR] [Nantes] [JulianBreaud] taxifare v1"

class Trainer():

### Code 
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    ### Code pour MLFLOW
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.mlflow_log_param('model', 'LinearRegression')
        return pipe

    def train(self, X_train, y_train):
        """set and train the pipeline"""
        '''returns a trained pipelined model'''
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(f'rmse : {rmse}')
        self.mlflow_log_metric('rmse', rmse)
        return rmse


if __name__ == "__main__":
    # get data / store the data in a DataFrame
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # build pipeline
    trainer = Trainer(X, y)
    pipeline = trainer.set_pipeline()
    # train
    trainer.train(X_train, y_train)

    # evaluate
    rmse = trainer.evaluate(X_val, y_val)
    print(f'rmse : {rmse}')
