import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from model import create_model,MultiOutputMSELoss,reset_random

DATA_DIR="Data/preprocessed"
MODEL_DIR="model"
EPOCHS=25
BATCH_SIZE=32
LEARNING_RATE=0.0001
SEED=42

def load_data():
    X_train=np.load(os.path.join(DATA_DIR,"X_train.npy"))
    y_train=np.load(os.path.join(DATA_DIR,"y_train.npy"))
    X_test=np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_test=np.load(os.path.join(DATA_DIR,"y_test.npy"))
    return torch.tensor(X_train,dtype=torch.float32),torch.tensor(y_train,dtype=torch.float32),torch.tensor(X_test,dtype=torch.float32),torch.tensor(y_test,dtype=torch.float32)

def train():
    reset_random(SEED)
    os.makedirs(MODEL_DIR,exist_ok=True)

    X_train,y_train,X_test,y_test=load_data()
    train_loader=DataLoader(TensorDataset(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(TensorDataset(X_test,y_test),batch_size=BATCH_SIZE,shuffle=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=create_model(input_dim=X_train.shape[1],hidden_dim=128,blocks=3,dropout=0.2).to(device)
    criterion=MultiOutputMSELoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

    history={"train_loss":[],"val_loss":[]}
    best_loss=float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss=0.0

        for X,y in train_loader:
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            pred=model(X)
            loss=criterion(pred,y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*X.size(0)

        train_loss/=len(train_loader.dataset)
        model.eval()
        val_loss=0.0

        with torch.no_grad():
            for X,y in test_loader:
                X,y=X.to(device),y.to(device)
                pred=model(X)
                loss=criterion(pred,y)
                val_loss+=loss.item()*X.size(0)
                
        val_loss/=len(test_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss<best_loss:
            best_loss=val_loss
            torch.save({"model_state_dict":model.state_dict(),"input_dim":X_train.shape[1],"hidden_dim":128,"blocks":3,"dropout":0.2,"epoch":epoch+1,"loss":val_loss},os.path.join(MODEL_DIR,"best_model.pth"))
    
    with open(os.path.join(MODEL_DIR,"history.json"),"w") as f:
        json.dump(history,f)

if __name__=="__main__":
    train()