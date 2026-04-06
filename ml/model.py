import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=10, n_features=4, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        
        # Hidden shape: (1, batch_size, hidden_dim)
        # We need to repeat the hidden state to match the sequence length
        hidden = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(hidden)
        
        # Reconstruct
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility for production inference.
        Returns the reconstruction error (MSE) per feature.
        """
        reconstruction = self.forward(x)
        # MSE across features
        # x shape: (1, seq_len, n_features) (batch_size=1 typically in realtime)
        error = torch.mean((x - reconstruction) ** 2, dim=1)
        return error
