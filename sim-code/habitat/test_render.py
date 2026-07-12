"""Minimal render test using proper config pipeline (no multiprocessing)."""
import numpy as np
import habitat_sim
from PIL import Image
import os, sys

SCENE = "data/scene_datasets/hm3d/hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"

# Build config using habitat-sim's own Configuration, not raw SimulatorConfiguration
# Mimic what HabitatSim.create_sim_config does
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE
sim_cfg.gpu_device_id = 0
sim_cfg.enable_physics = False

agent_cfg = habitat_sim.AgentConfiguration()
sensor_spec = habitat_sim.SensorSpec()
sensor_spec.uuid = "rgb"
sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
sensor_spec.resolution = [480, 640]
sensor_spec.position = [0.0, 0.88, 0.0]
sensor_spec.hfov = 79
agent_cfg.sensor_specifications = [sensor_spec]
agent_cfg.height = 0.88
agent_cfg.radius = 0.1

# Build the final Configuration
cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

light_modes = {
    "no_lights": habitat_sim.gfx.NO_LIGHT_KEY,
    "default":   habitat_sim.gfx.DEFAULT_LIGHTING_KEY,
}
results = {}

for label, light_key in light_modes.items():
    try:
        sim_cfg.scene_light_setup = light_key
        print(f"\n--- {label}: scene_light_setup={light_key!r} ---")
        sim = habitat_sim.Simulator(cfg)
        obs = sim.get_sensor_observations()
        rgb = obs['rgb']
        mean = rgb.mean()
        uniq = len(np.unique(rgb))
        print(f"RGB: shape={rgb.shape}, min={rgb.min()}, max={rgb.max()}, mean={mean:.2f}, unique={uniq}")

        fname = f"test_render_{label}.png"
        Image.fromarray(rgb).save(fname)
        print(f"Saved {fname}")

        # Save as NPY for raw analysis
        np.save(f"test_render_{label}.npy", rgb)

        results[label] = rgb
        sim.close()
    except Exception as e:
        import traceback
        traceback.print_exc()

print("\n=== SUMMARY ===")
for label, arr in results.items():
    print(f"{label:12s}: mean={arr.mean():5.1f}  min={arr.min():3d}  max={arr.max():3d}")
