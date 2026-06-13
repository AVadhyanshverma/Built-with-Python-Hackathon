import os
import torch
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 1. THE PHOTO-Z ARCHITECTURE ---
class PhotoZNet(nn.Module):
    def __init__(self):
        super(PhotoZNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.Mish(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.Mish(), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

def generate_cosmic_depth_map(csv_file):
    print("==================================================")
    print("      DEEP FIELD BULK INFERENCE & MAPPING         ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model and Scaler
    model = PhotoZNet().to(device)
    try:
        model.load_state_dict(torch.load("models/quasar_photoz_mlp.pth", map_location=device))
        model.eval()
        scaler = joblib.load("models/photoz_scaler.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] Missing file: {e}")
        return

    # 2. Load the CSV Data
    try:
        print(f"[DATA] Loading targets from {csv_file}...")
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_file}")
        return

    # 3. Feature Engineering (Calculating the 4 colors for the whole dataset at once)
    print(f"[AI] Processing {len(df)} celestial targets through the neural network...")
    u, g, r, i, z = df['u'].values, df['g'].values, df['r'].values, df['i'].values, df['z'].values
    
    # Combine into 9 features (u, g, r, i, z, u-g, g-r, r-i, i-z)
    raw_features = np.column_stack((u, g, r, i, z, u-g, g-r, r-i, i-z))
    
    # Scale and convert to Tensor
    scaled_features = scaler.transform(raw_features)
    tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # 4. Bulk Inference!
    with torch.no_grad():
        predictions = model(tensor_data).cpu().numpy().flatten()
    
    # Save predictions back to the dataframe
    df['Predicted_Redshift_Z'] = predictions
    
    output_csv = "mapped_" + csv_file
    df.to_csv(output_csv, index=False)
    print(f"✅ [SUCCESS] Bulk inference complete. Saved to {output_csv}")

    # ==========================================
    # 5. GENERATE THE COSMIC DEPTH MAP IMAGE
    # ==========================================
    print("[SYSTEM] Rendering Cosmic Depth Map...")
    
    plt.style.use('dark_background') # Make it look like space
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot: X=RA, Y=Dec, Color=Predicted Redshift, Size=Brightness(inverted 'i' band)
    # We use the 'magma' colormap: yellow/white = close, purple/black = ancient/far away
    scatter = ax.scatter(
        df['ra'], df['dec'], 
        c=df['Predicted_Redshift_Z'], 
        cmap='magma_r', 
        s=(26 - df['i']) ** 2, # Brighter objects look bigger
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

    # Aesthetics
    cbar = plt.colorbar(scatter)
    cbar.set_label('AI Predicted Redshift (Distance)', fontsize=12, color='white')
    
    ax.set_title("AI-Generated Cosmic Depth Map (SMACS 0723 Region)", fontsize=16, fontweight='bold', color='white')
    ax.set_xlabel("Right Ascension (RA)", fontsize=12)
    ax.set_ylabel("Declination (Dec)", fontsize=12)
    ax.grid(True, alpha=0.1)

    # Invert X axis (astronomy standard for RA)
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig("cosmic_depth_map.png", dpi=300, facecolor='black')
    print("🌌 [RENDER] Image saved successfully as 'cosmic_depth_map.png'")
    plt.show()

if __name__ == "__main__":
    generate_cosmic_depth_map("deep_field_targets_RA161.3.csv")
