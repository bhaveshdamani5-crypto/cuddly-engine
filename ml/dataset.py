import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class EnvironmentalDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len]

def generate_synthetic_data(num_samples=10000, anomaly_fraction=0.05):
    """
    Generates synthetic multivariate time-series data for environmental sensors.
    Features: [Temperature, Humidity, CO2, PM2.5]
    """
    time = np.linspace(0, 100, num_samples)
    
    # Normal seasonal variations
    temp = 25 + 5 * np.sin(time) + np.random.normal(0, 0.5, num_samples)
    humidity = 40 + 10 * np.cos(time * 0.5) + np.random.normal(0, 1, num_samples)
    co2 = 400 + 20 * np.sin(time * 0.1) + np.random.normal(0, 5, num_samples)
    pm25 = 15 + 5 * np.sin(time * 2) + np.random.normal(0, 2, num_samples)
    
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'co2': co2,
        'pm25': pm25,
        'is_anomaly': 0
    })
    
    # Inject anomalies (e.g., sudden spikes representing leaks, fires, or pollution events)
    num_anomalies = int(num_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(num_samples - 10, num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['fire', 'leak', 'smog'])
        if anomaly_type == 'fire':
            df.loc[idx:idx+5, 'temperature'] += np.random.uniform(10, 20)
            df.loc[idx:idx+5, 'co2'] += np.random.uniform(200, 500)
            df.loc[idx:idx+5, 'pm25'] += np.random.uniform(50, 150)
        elif anomaly_type == 'leak':
            df.loc[idx:idx+5, 'co2'] += np.random.uniform(300, 800)
        elif anomaly_type == 'smog':
            df.loc[idx:idx+10, 'pm25'] += np.random.uniform(100, 200)
        
        df.loc[idx:idx+5, 'is_anomaly'] = 1

    return df

def get_dataloaders(seq_len=10, batch_size=32):
    df = generate_synthetic_data()
    
    # Split normal vs anomaly for training
    # Autoencoders train only on normal data
    normal_data = df[df['is_anomaly'] == 0].drop(columns=['is_anomaly'])
    
    scaler = StandardScaler()
    scaled_normal = scaler.fit_transform(normal_data)
    
    dataset = EnvironmentalDataset(scaled_normal, seq_len=seq_len)
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler
