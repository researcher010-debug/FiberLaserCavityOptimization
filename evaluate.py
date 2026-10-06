import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from model import create_model,reset_random

DATA_DIR="Data/preprocessed"
MODEL_DIR="model"
RESULT_DIR="results"
TARGET_NAMES=["T_out","BW","P_peak"]
SEED=42

def load_data():
    X_train=np.load(os.path.join(DATA_DIR,"X_train.npy"))
    X_test=np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_train=np.load(os.path.join(DATA_DIR,"y_train.npy"))
    y_test=np.load(os.path.join(DATA_DIR,"y_test.npy"))
    x_scaler=joblib.load(os.path.join(DATA_DIR,"x_scaler.pkl"))
    y_scaler=joblib.load(os.path.join(DATA_DIR,"y_scaler.pkl"))
    return X_train,X_test,y_train,y_test,x_scaler,y_scaler

def calculate_metrics(actual,predicted):
    results=[]
    for i,name in enumerate(TARGET_NAMES):
        y_true=actual[:,i]
        y_pred=predicted[:,i]
        mse=mean_squared_error(y_true,y_pred)
        mape=np.mean(np.abs((y_true-y_pred)/np.maximum(np.abs(y_true),1e-8)))*100
        results.append({"Target":name,"MSE":mse,"RMSE":np.sqrt(mse),"MAE":mean_absolute_error(y_true,y_pred),"MAPE":mape,"R2":r2_score(y_true,y_pred)})
    return results

def predict(model,X,device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X,dtype=torch.float32).to(device)).cpu().numpy()

def plot_loss(history):
    epochs=np.arange(1,len(history["train_loss"])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs,history["train_loss"],label="Train Loss")
    plt.plot(epochs,history["val_loss"],label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"loss_curve.png"),dpi=300)
    plt.close()

def plot_actual_vs_predicted(actual,predicted,split):
    for i,name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(7,5))
        plt.plot(actual[:,i],label="Actual",linewidth=2)
        plt.plot(predicted[:,i],label="Predicted",linewidth=1.5,linestyle="--")
        plt.xlabel("Sample Index")
        plt.ylabel(name)
        plt.title(f"{split}: Actual vs Predicted - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR,f"{split.lower()}_{name}_actual_vs_predicted.png"),dpi=300)
        plt.close()

def evaluate():
    reset_random(SEED)
    os.makedirs(RESULT_DIR,exist_ok=True)

    X_train,X_test,y_train,y_test,x_scaler,y_scaler=load_data()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(os.path.join(MODEL_DIR,"best_model.pth"),map_location=device,weights_only=False)
    model=create_model(input_dim=checkpoint["input_dim"],hidden_dim=checkpoint["hidden_dim"],blocks=checkpoint["blocks"],dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    train_pred=predict(model,X_train,device)
    test_pred=predict(model,X_test,device)
    y_train_actual=y_scaler.inverse_transform(y_train)
    y_test_actual=y_scaler.inverse_transform(y_test)
    train_pred_actual=y_scaler.inverse_transform(train_pred)
    test_pred_actual=y_scaler.inverse_transform(test_pred)
    metrics=calculate_metrics(y_train_actual,train_pred_actual)+calculate_metrics(y_test_actual,test_pred_actual)

    for i,item in enumerate(metrics):
        item["Split"]="Train" if i<len(TARGET_NAMES) else "Test"
    pd.DataFrame(metrics)[["Split","Target","MSE","RMSE","MAE","MAPE","R2"]].to_csv(os.path.join(RESULT_DIR,"metrics.csv"),index=False)
    pd.DataFrame({"Split":["Train"]*len(y_train_actual)+["Test"]*len(y_test_actual),"T_out_actual":np.concatenate([y_train_actual[:,0],y_test_actual[:,0]]),"T_out_predicted":np.concatenate([train_pred_actual[:,0],test_pred_actual[:,0]]),"BW_actual":np.concatenate([y_train_actual[:,1],y_test_actual[:,1]]),"BW_predicted":np.concatenate([train_pred_actual[:,1],test_pred_actual[:,1]]),"P_peak_actual":np.concatenate([y_train_actual[:,2],y_test_actual[:,2]]),"P_peak_predicted":np.concatenate([train_pred_actual[:,2],test_pred_actual[:,2]])}).to_csv(os.path.join(RESULT_DIR,"predictions.csv"),index=False)
    
    with open(os.path.join(MODEL_DIR,"history.json")) as f:
        history=json.load(f)

    plot_loss(history)
    plot_actual_vs_predicted(y_train_actual,train_pred_actual,"Train")
    plot_actual_vs_predicted(y_test_actual,test_pred_actual,"Test")
    print(pd.DataFrame(metrics)[["Split","Target","MSE","RMSE","MAE","MAPE","R2"]].to_string(index=False))

if __name__=="__main__":
    evaluate()