import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_PATH="Data/raw/dataset.csv"
OUTPUT_DIR="Data/preprocessed"

INPUT_COLUMNS=["beta_2","gamma","L","P_in","T_in"]
TARGET_COLUMNS=["T_out","BW","P_peak"]

def load_data():
    df=pd.read_csv(DATA_PATH)
    df=df.dropna()
    df=df.drop_duplicates()
    return df

def preprocess():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    df=load_data()

    X=df[INPUT_COLUMNS].astype(np.float32).values
    y=df[TARGET_COLUMNS].astype(np.float32).values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42,shuffle=True)

    x_scaler=MinMaxScaler()
    y_scaler=MinMaxScaler()

    X_train=x_scaler.fit_transform(X_train).astype(np.float32)
    X_test=x_scaler.transform(X_test).astype(np.float32)
    y_train=y_scaler.fit_transform(y_train).astype(np.float32)
    y_test=y_scaler.transform(y_test).astype(np.float32)

    np.save(os.path.join(OUTPUT_DIR,"X_train.npy"),X_train)
    np.save(os.path.join(OUTPUT_DIR,"X_test.npy"),X_test)
    np.save(os.path.join(OUTPUT_DIR,"y_train.npy"),y_train)
    np.save(os.path.join(OUTPUT_DIR,"y_test.npy"),y_test)
    
    joblib.dump(x_scaler,os.path.join(OUTPUT_DIR,"x_scaler.pkl"))
    joblib.dump(y_scaler,os.path.join(OUTPUT_DIR,"y_scaler.pkl"))

if __name__=="__main__":
    preprocess()