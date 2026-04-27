import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
def load_data(disturb=False):
    print("🔄 Loading dataset...")
    df = pd.read_hdf("data/metr-la.h5")
    print("✅ Dataset:", df.shape)

    data = df.values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    data = data[:6000]

    if disturb:
        print("⚠️ Disturbance applied")
        data[2000:2100] *= 0.5

    return data


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len=8):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)


def split_data(X, Y):
    s1 = int(0.7 * len(X))
    s2 = int(0.85 * len(X))
    return X[:s1], Y[:s1], X[s1:s2], Y[s1:s2], X[s2:], Y[s2:]


# =========================
# LOAD GRAPH
# =========================
with open("data/adj_mx.pkl", "rb") as f:
    _, _, adj = pickle.load(f, encoding='latin1')

edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long).to(device)
num_nodes = adj.shape[0]


# =========================
# IMPROVED STGCN (FINAL)
# =========================
class STGCN(nn.Module):
    def __init__(self, num_nodes):
        super(STGCN, self).__init__()

        self.gcn1 = GCNConv(1, 16)
        self.gcn2 = GCNConv(16, 16)

        self.dropout = nn.Dropout(0.4)

        # 🔥 stronger temporal modeling
        self.lstm = nn.LSTM(num_nodes * 16, 128, batch_first=True)

        self.fc = nn.Linear(128, num_nodes)

    def forward(self, x):
        B, T, N = x.shape

        residual = x  # 🔥 residual connection

        x = x.reshape(B*T, N)
        x = x.unsqueeze(-1)

        x = torch.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)

        x = torch.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)

        x = x.reshape(B, T, -1)

        lstm_out, _ = self.lstm(x)

        out = self.fc(lstm_out[:, -1, :])

        # 🔥 residual learning
        out = out + residual[:, -1, :]

        return out


# =========================
# LSTM BASELINE
# =========================
class LSTMModel(nn.Module):
    def __init__(self, num_nodes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_nodes, 64, batch_first=True)
        self.fc = nn.Linear(64, num_nodes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================
# TRAIN
# =========================
def train_model(model, X_train, Y_train, X_val, Y_val, name):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    loss_fn = nn.MSELoss()

    batch_size = 32

    print(f"\n🚀 Training {name}...\n")

    for epoch in range(12):

        model.train()
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = Y_train[i:i+batch_size]

            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val[:128])
            val_loss = loss_fn(val_out, Y_val[:128])

        print(f"{name} Epoch {epoch+1} | Train: {total_loss:.4f} | Val: {val_loss.item():.4f}")

    return model


# =========================
# EVALUATION
# =========================
def evaluate(model, X_test, Y_test):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            preds.append(model(X_test[i:i+32]).cpu())

    preds = torch.cat(preds).numpy()
    true = Y_test.cpu().numpy()

    mae = mean_absolute_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))

    return mae, rmse, preds, true


# =========================
# SMART OPTIMIZATION (FIXED)
# =========================
def optimize_traffic(pred, adj_matrix):

    print("\n🚦 TRAFFIC OPTIMIZATION ENGINE")

    avg = pred.mean(axis=0)

    # 🔥 dynamic threshold (VERY IMPORTANT FIX)
    threshold = np.percentile(avg, 20)

    congestion_nodes = np.where(avg < threshold)[0]

    for node in congestion_nodes[:5]:

        neighbors = np.where(adj_matrix[node] > 0)[0]

        if len(neighbors) == 0:
            continue

        best = neighbors[np.argmax(avg[neighbors])]

        print(f"\n🚨 Sensor {node} congested")
        print(f"➡️ Best reroute → Sensor {best}")
        print("🚦 Action → Increase GREEN time")

    return congestion_nodes


# =========================
# DIGITAL TWIN (REAL FEEDBACK)
# =========================
def simulate_real_time(model, X_test):

    print("\n🌐 Running Digital Twin Simulation...")

    current_state = X_test[0:1].clone()

    for i in range(5):

        pred = model(current_state).cpu().detach().numpy()

        print(f"\n⏱️ Cycle {i+1}")

        congested = optimize_traffic(pred, adj)

        # 🔥 stronger feedback loop
        for node in congested:
            current_state[:, :, node] += 0.3

        time.sleep(1)


# =========================
# RUN
# =========================
def run(disturb=False):

    data = load_data(disturb)
    X, Y = create_sequences(data)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X, Y)

    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float).to(device)

    stgcn = train_model(STGCN(num_nodes), X_train, Y_train, X_val, Y_val, "STGCN")
    torch.save(stgcn.state_dict(), "STGCN.pth")
    lstm = train_model(LSTMModel(num_nodes), X_train, Y_train, X_val, Y_val, "LSTM")

    stgcn_mae, stgcn_rmse, pred, true = evaluate(stgcn, X_test, Y_test)
    lstm_mae, lstm_rmse, _, _ = evaluate(lstm, X_test, Y_test)

    print("\n📊 FINAL RESULTS")
    print("STGCN:", stgcn_mae, stgcn_rmse)
    print("LSTM :", lstm_mae, lstm_rmse)

    simulate_real_time(stgcn, X_test)

    plt.figure()
    plt.plot(true[:, 0], label="Actual")
    plt.plot(pred[:, 0], label="Predicted")
    plt.legend()
    plt.title("Traffic Prediction (Digital Twin)")
    plt.show()


# =========================
# MAIN
# =========================
print("\n===== NORMAL =====")
run(False)

print("\n===== DISTURBED =====")
run(True)