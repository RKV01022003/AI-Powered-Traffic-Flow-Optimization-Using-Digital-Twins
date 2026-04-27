import streamlit as st
import numpy as np
import torch
import pickle
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

# Try Plotly (fallback safe)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Traffic AI", layout="wide")

st.markdown("# 🚦 Smart Traffic AI Dashboard")
st.markdown("### AI-Based Traffic Prediction • Optimization • Digital Twin")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LANGUAGE SUPPORT
# =========================
lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi", "Hinglish"])

def T(en, hi, hn):
    return {"English": en, "Hindi": hi, "Hinglish": hn}[lang]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")

refresh_interval = st.sidebar.slider(
    T("Refresh Interval (sec)", "रिफ्रेश समय", "Refresh time"),
    2, 15, 5
)

# =========================
# AUTO REFRESH (SAFE)
# =========================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =========================
# LOAD GRAPH
# =========================
@st.cache_resource
def load_graph():
    with open("data/adj_mx.pkl", "rb") as f:
        _, _, adj = pickle.load(f, encoding='latin1')
    return adj

adj = load_graph()
num_nodes = adj.shape[0]

edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long).to(device)

# =========================
# MODEL
# =========================
class STGCN(torch.nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.gcn1 = GCNConv(1, 16)
        self.gcn2 = GCNConv(16, 16)
        self.lstm = torch.nn.LSTM(num_nodes * 16, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, num_nodes)

    def forward(self, x):
        B, T, N = x.shape
        x = x.reshape(B*T, N)
        x = x.unsqueeze(-1)

        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))

        x = x.reshape(B, T, -1)
        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1, :])

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = STGCN(num_nodes)
    model.load_state_dict(torch.load("STGCN.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_hdf("data/metr-la.h5")
    data = df.values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data[:6000]

data = load_data()

# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(data, seq_len=8):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

X, Y = create_sequences(data)

split = int(0.85 * len(X))
X_test = X[split:]
Y_test = Y[split:]

X_test = torch.tensor(X_test, dtype=torch.float).to(device)

# =========================
# SAFE INDEX
# =========================
max_index = len(X_test) - 1

sample_index = st.sidebar.slider(
    T("Select Time Step", "समय चुनें", "Time select karo"),
    0,
    max_index,
    0
)

# =========================
# PREDICTION
# =========================
sample = X_test[sample_index:sample_index+1]

with torch.no_grad():
    pred = model(sample).cpu().numpy()[0]

actual = Y_test[sample_index]
error = np.abs(actual - pred)

# =========================
# KPI CARDS
# =========================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{np.mean(error):.4f}")
col2.metric("RMSE", f"{np.sqrt(np.mean(error**2)):.4f}")
col3.metric("🚨 Congestion", int(np.sum(pred < np.percentile(pred, 20))))

with st.expander("📖 Explanation"):
    st.write(T(
        "MAE = average error, RMSE = penalizes large errors more.",
        "MAE औसत त्रुटि है, RMSE बड़ी त्रुटियों को ज्यादा महत्व देता है।",
        "MAE average error hai aur RMSE bade error ko zyada punish karta hai."
    ))

# =========================
# GRAPH
# =========================
st.subheader("📈 Prediction vs Actual")

df_chart = pd.DataFrame({"Actual": actual, "Predicted": pred})

st.line_chart(df_chart)

# =========================
# ERROR DISTRIBUTION
# =========================
st.subheader("📉 Error Distribution")

if PLOTLY_AVAILABLE:
    fig = px.histogram(x=error, nbins=40)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.bar_chart(error)

# =========================
# CONGESTION ANALYSIS
# =========================
st.subheader("🚨 Congestion Analysis")

threshold = np.percentile(pred, 20)
congested_idx = np.where(pred < threshold)[0]

if len(congested_idx) == 0:
    st.success(T("No congestion detected", "कोई जाम नहीं", "No jam"))
else:
    top_nodes = congested_idx[:10]

    if PLOTLY_AVAILABLE:
        fig2 = px.bar(x=top_nodes, y=pred[top_nodes])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.bar_chart(pred[top_nodes])

# =========================
# OPTIMIZATION ENGINE
# =========================
st.subheader("🚦 Traffic Optimization")

for node in congested_idx[:5]:
    neighbors = np.where(adj[node] > 0)[0]

    if len(neighbors) > 0:
        best = neighbors[np.argmax(pred[neighbors])]
        st.warning(f"{T('Sensor', 'सेंसर', 'Sensor')} {node} → {T('Reroute to', 'रास्ता बदलें', 'Reroute')} {best}")
    else:
        st.warning(f"Sensor {node} → No route")

# =========================
# HEATMAP
# =========================
st.subheader("🔥 Traffic Heatmap")

heatmap_df = pd.DataFrame(pred.reshape(1, -1))
st.dataframe(heatmap_df)

# =========================
# INSIGHTS
# =========================
st.subheader("📌 AI Insights")

st.info(T(
    f"Model detected {len(congested_idx)} congested nodes.",
    f"मॉडल ने {len(congested_idx)} जाम वाले स्थान पहचाने।",
    f"Model ne {len(congested_idx)} jam detect kiya."
))

# =========================
# FOOTER
# =========================
st.markdown("""
---
### 🚀 Features
- STGCN Model (Graph + Time)
- Real-Time Prediction
- Congestion Detection
- Smart Signal Optimization
- Explainable AI Dashboard
""")