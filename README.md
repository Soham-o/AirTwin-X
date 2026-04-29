# AirTwin — Smart City Air Quality Digital Twin

A professional Streamlit command-center dashboard combining three AI layers:
**Cleanest Path Navigator · Causal XAI Engine · Visual CV Proxy**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` and is immediately runnable — no API keys or external data feeds required. All data is generated via a physics-inspired mock engine.

## Architecture: Three AI Layers

### Layer 1 — Cleanest Path Navigator
- **Routing engine**: Inverse-distance weighted AQI interpolation across sensor nodes
- **Risk profiling**: Sensitive/Asthmatic users incur a 25% AQI penalty on high-exposure segments
- **Output**: Ranked routes (Fastest / Balanced / Cleanest) with exposure score

### Layer 2 — Causal Digital Twin (XAI)
- **Dispersion model**: Gaussian plume approximation using wind vector + source intensity
- **Attribution**: SHAP-style % breakdown — "75% of this spike is from Rohini Power Plant due to NW winds"
- **Interactive**: Wind direction/speed sliders update attribution in real-time

### Layer 3 — Visual CV Proxy
- **Mock camera feed**: Procedurally generated haze image (haze depth scales with AQI)
- **CV estimate**: Simulates CNN model output with realistic noise
- **Fusion gate**: CV confidence < 70% → weight reduced to 0.10

## Late Fusion Formula

```
Final_AQI = 0.70 × Sensor_AQI + 0.20 × Visual_CV_AQI + 0.10 × Causal_Prior
```

Weights are static here; in production these are learned via Bayesian optimization on hourly ground-truth validation data from CPCB monitoring stations.

## Extending to Production

| Component | Swap-in |
|-----------|---------|
| Sensor mock | CPCB / OpenAQ REST API |
| CV model | MobileNetV3 fine-tuned on Delhi cam frames |
| Routing | OpenRouteService or GraphHopper with AQI edge weights |
| SHAP | Real `shap` library on a trained XGBoost/LightGBM model |
| Real-time | Kafka stream → Streamlit `st.rerun()` |
