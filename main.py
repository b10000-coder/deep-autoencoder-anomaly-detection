import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
import os

# Create results folder if not exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Data Generation
# ---------------------------------------------------------

def generate_signals(num=1000, setup=None):
    """
    Generates normal and anomalous 1D signals based on sine combinations.
    
    setup = {
        "A_normal":  (-2, 2),
        "f_normal":  (-2, 2),
        "A_anom":    [(2,5), (-5,-2)],
        "f_anom":    [(5,10), (-10,-5)]
    }
    """

    if setup is None:
        setup = {
            "A_normal":  (-2, 2),
            "f_normal":  (-2, 2),
            "A_anom":    [(2,5), (-5,-2)],
            "f_anom":    [(5,10), (-10,-5)]
        }

    x = np.linspace(-8*np.pi, 8*np.pi, 1024)
    normals = []
    anomalies = []

    # Helper for picking from two disjoint ranges
    def pick_from_ranges(ranges):
        r = ranges[np.random.randint(0, len(ranges))]
        return np.random.uniform(r[0], r[1])

    for _ in range(num):
        # === Normal Signal ===
        A = np.random.uniform(*setup["A_normal"], size=3)
        f = np.random.uniform(*setup["f_normal"], size=3)

        y = A[0]*np.sin(f[0]*x) + A[1]*np.sin(f[1]*x) + A[2]*np.sin(f[2]*x)
        normals.append(y)

        # === Anomalous Signal ===
        A_anom = pick_from_ranges(setup["A_anom"])
        f_anom = pick_from_ranges(setup["f_anom"])

        y_a = y + A_anom * np.sin(f_anom * x)
        anomalies.append(y_a)

    return np.array(normals), np.array(anomalies), x


# ---------------------------------------------------------
# 2. Build the Autoencoder (1D-CNN)
# ---------------------------------------------------------

def build_autoencoder(input_len=1024):
    inp = layers.Input(shape=(input_len, 1))

    # Encoder
    x = layers.Conv1D(16, 7, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, 7, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2)(x)

    # Decoder
    x = layers.Conv1D(8, 7, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)

    out = layers.Conv1D(1, 7, activation='linear', padding='same')(x)

    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')

    return model


# ---------------------------------------------------------
# 3. Train + Compute Threshold
# ---------------------------------------------------------

def train_autoencoder(normals):
    """Train the autoencoder only on NORMAL signals."""
    X = normals.reshape((-1, normals.shape[1], 1))

    model = build_autoencoder(normals.shape[1])
    model.fit(X, X, epochs=20, batch_size=32, verbose=1)

    # Reconstruction MSE for thresholding
    X_pred = model.predict(X)
    mse = np.mean((X_pred - X)**2, axis=(1,2))

    threshold = np.percentile(mse, 99)  # 0.99 percentile
    return model, threshold, mse


# ---------------------------------------------------------
# 4. Anomaly Classification
# ---------------------------------------------------------

def classify(model, data, threshold):
    X = data.reshape((-1, data.shape[1], 1))
    pred = model.predict(X)
    mse = np.mean((pred - X)**2, axis=(1,2))
    labels = (mse > threshold).astype(int)
    return labels, mse


# ---------------------------------------------------------
# 5. Run Everything At Once (Like the Article)
# ---------------------------------------------------------

def run_all(num=1000, setup=None, plot_examples=True):
    normals, anomalies, x = generate_signals(num=num, setup=setup)

    model, thr, mse_norm_train = train_autoencoder(normals)

    # Classify anomalies and normals
    labels_norm, mse_norm = classify(model, normals, thr)
    labels_anom, mse_anom = classify(model, anomalies, thr)

    # Compute metrics
    fp = labels_norm.sum()             # normal -> anomaly
    tp = labels_anom.sum()             # anomaly -> anomaly detected
    fn = len(anomalies) - tp           # anomaly -> normal (missed)

    # ROC AUC
    all_mse = np.concatenate([mse_norm, mse_anom])
    all_labels = np.array([0]*len(mse_norm) + [1]*len(mse_anom))
    auc = roc_auc_score(all_labels, all_mse)

    print("\n--- Results ---")
    print("Threshold (99th percentile):", thr)
    print("False Positives:", fp)
    print("True Positives:", tp)
    print("False Negatives:", fn)
    print("ROC AUC:", auc)

    # ---------------------------------------------------
    # Save results in text file
    # ---------------------------------------------------
    results_text = (
        f"Threshold: {thr}\n"
        f"False Positives: {fp}\n"
        f"True Positives: {tp}\n"
        f"False Negatives: {fn}\n"
        f"ROC AUC: {auc}\n"
    )
    with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
        f.write(results_text)

    # ---------------------------------------------------
    # Save plot
    # ---------------------------------------------------
    if plot_examples:
        import matplotlib.gridspec as gridspec

    # ----------------------------------------------------
    # 1) Create multi-panel figure
    # ----------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # === Subplot A: Normal distribution (zoom + KDE) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Normal MSE Distribution (Zoomed, KDE)")
    sns.histplot(mse_norm, bins=50, kde=True, color="blue", ax=ax1)
    ax1.axvline(thr, color='black', linestyle='--', label="Threshold")
    ax1.set_xlim(0, np.percentile(mse_norm, 99))  # zoom into normal-only region
    ax1.legend()

    # === Subplot B: Anomaly distribution (KDE) ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Anomalous MSE Distribution (KDE)")
    sns.histplot(mse_anom, bins=50, kde=True, color="red", ax=ax2)
    ax2.axvline(thr, color='black', linestyle='--', label="Threshold")
    ax2.legend()

    # === Subplot C: Overlay with log-scale ========
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("Overlay (Log Scale) — Normal vs Anomalous Reconstruction Error")
    sns.histplot(mse_norm, bins=50, kde=True, color="blue", alpha=0.6, label="Normal", ax=ax3)
    sns.histplot(mse_anom, bins=50, kde=True, color="red", alpha=0.6, label="Anomalous", ax=ax3)
    ax3.axvline(thr, color='black', linestyle='--', label="Threshold")
    ax3.set_xscale("log")   # log x-axis to reveal separation
    ax3.legend()

    plt.tight_layout()

    # Save it
    save_path = os.path.join(RESULTS_DIR, "reconstruction_error_panels.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] {save_path}")


    return {
        "model": model,
        "threshold": thr,
        "mse_norm": mse_norm,
        "mse_anom": mse_anom
    }



if __name__ == "__main__":
    run_all()
