import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
from .model import LSTMAutoencoder
from .dataset import get_dataloaders

def train_model():
    print("Initializing training...")
    seq_len = 10
    n_features = 4
    epochs = 20
    
    train_loader, val_loader, scaler = get_dataloaders(seq_len=seq_len)
    
    # Save the scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(seq_len=seq_len, n_features=n_features).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save Model
    # We export as TorchScript for Edge deployment / easier inference
    print("Exporting model to TorchScript...")
    model.eval()
    example_input = torch.randn(1, seq_len, n_features).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('models/anomaly_detector.pt')
    
    # Save a generic PyTorch dict as well
    torch.save(model.state_dict(), 'models/anomaly_detector_weights.pth')
    
    print("Training complete. Models saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()
