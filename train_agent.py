import osmnx as ox
import networkx as nx
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

# ─────────────────────────────────────────────
# CONSTANTS — must match app.py exactly
# ─────────────────────────────────────────────
MAX_NEIGHBORS = 8  # covers real Delhi intersections (typically 5-8 exits)
OBS_SIZE = 4 + MAX_NEIGHBORS  # [curr_lat, curr_lon, targ_lat, targ_lon, aqi_0..aqi_7]

# ─────────────────────────────────────────────
# GRAPH ACQUISITION
# ─────────────────────────────────────────────
GRAPH_FILE = "delhi.graphml"

if os.path.exists(GRAPH_FILE):
    print(f"Loading cached graph from {GRAPH_FILE}...")
    G = ox.load_graphml(GRAPH_FILE)
else:
    print("Downloading Delhi road network (5km radius)...")
    point = (28.6139, 77.2090)
    G = ox.graph_from_point(point, dist=5000, network_type='drive')

    # Keep only the largest weakly-connected component to guarantee reachability
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_wcc).copy()

    ox.save_graphml(G, GRAPH_FILE)
    print(f"Graph saved: {len(G.nodes)} nodes, {len(G.edges)} edges.")

# Inject synthetic AQI that mimics real Delhi patterns:
#   - Primary roads get higher baseline pollution
#   - Random noise models micro-level variability
ROAD_AQI_BASELINE = {
    'motorway': 280, 'trunk': 250, 'primary': 220,
    'secondary': 180, 'tertiary': 140, 'residential': 100,
}
for u, v, k, data in G.edges(keys=True, data=True):
    highway = data.get('highway', 'residential')
    if isinstance(highway, list):
        highway = highway[0]
    base = ROAD_AQI_BASELINE.get(highway, 120)
    data['mock_aqi'] = float(np.clip(base + np.random.normal(0, 30), 30, 450))

print(f"AQI injected on {len(G.edges())} edges.")

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
class RealCleanAirEnv(gym.Env):
    """
    RL environment for AQI-aware navigation on a real OSMnx road graph.

    Action space  : Discrete(MAX_NEIGHBORS) — pick one of up to 8 outgoing edges.
    Observation   : [curr_lat, curr_lon, targ_lat, targ_lon, aqi_0 … aqi_7]
                    Padding with 0 when a node has fewer than MAX_NEIGHBORS exits.
    Reward shaping:
        • Travel cost   : proportional to edge length
        • AQI penalty   : non-linear (exponential) to strongly prefer clean corridors
        • Invalid move  : -50 (chosen action index ≥ actual neighbor count)
        • Goal bonus    : +500 on reaching target
    """

    metadata = {"render_modes": []}

    def __init__(self, graph, start_node, target_node):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.start_node = start_node
        self.target_node = target_node
        self.current_node = start_node

        self.action_space = spaces.Discrete(MAX_NEIGHBORS)
        self.observation_space = spaces.Box(
            low=-180.0, high=1000.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _get_obs(self):
        neighbors = list(self.graph.neighbors(self.current_node))
        curr_y = self.graph.nodes[self.current_node]['y']
        curr_x = self.graph.nodes[self.current_node]['x']
        targ_y = self.graph.nodes[self.target_node]['y']
        targ_x = self.graph.nodes[self.target_node]['x']

        neighbor_aqis = []
        for i in range(MAX_NEIGHBORS):
            if i < len(neighbors):
                edge_data = self.graph.get_edge_data(self.current_node, neighbors[i])
                # MultiDiGraph may have multiple parallel edges; take the first.
                edge_data = edge_data[0] if edge_data else {}
                neighbor_aqis.append(float(edge_data.get('mock_aqi', 150.0)))
            else:
                neighbor_aqis.append(0.0)  # padding

        return np.array([curr_y, curr_x, targ_y, targ_x] + neighbor_aqis, dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))

        if action < len(neighbors):
            next_node = neighbors[int(action)]
            edge_data = self.graph.get_edge_data(self.current_node, next_node)
            edge_data = edge_data[0] if edge_data else {}

            length = float(edge_data.get('length', 10.0))
            aqi = float(edge_data.get('mock_aqi', 150.0))

            # Non-linear penalty: AQI > 200 is exponentially worse
            travel_cost = length / 10.0
            aqi_penalty = (aqi / 100.0) ** 2
            reward = -(travel_cost + aqi_penalty)

            self.current_node = next_node
        else:
            reward = -50.0  # invalid move — action index beyond available neighbors

        done = self.current_node == self.target_node
        if done:
            reward += 500.0

        return self._get_obs(), reward, done, False, {}

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomise both start AND target each episode so the agent generalises
        # to arbitrary city-wide origin-destination pairs, not just one fixed pair.
        sampled = self.np_random.choice(len(self.nodes), size=2, replace=False)
        self.current_node = self.nodes[sampled[0]]
        self.target_node = self.nodes[sampled[1]]
        return self._get_obs(), {}


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
nodes = list(G.nodes())
env = RealCleanAirEnv(G, nodes[0], nodes[-1])

print(f"\nAction space : {env.action_space}")
print(f"Obs shape    : {env.observation_space.shape}")
print("Initialising PPO agent...\n")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# 50k steps for a quick test; use 500k+ for production-quality routing.
TIMESTEPS = 50_000
print(f"Training for {TIMESTEPS:,} timesteps...")
model.learn(total_timesteps=TIMESTEPS)

model.save("clean_air_agent")
print("\n✅ Agent saved as clean_air_agent.zip — ready for Streamlit.")