# navisim/tasks/navisim_warehouse_task.py

import torch
import isaaclab.sim as sim_utils
import imageio

from isaaclab.scene import InteractiveScene
from scene.warehouse_scene_cfg import NavisimWarehouseSceneCfg

WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

class NavigationTask:
    def __init__(self, num_envs=1, device="cuda:0", max_episode_len=500):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_len = max_episode_len

        sim_cfg = sim_utils.SimulationCfg(device=device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim_dt = self.sim.get_physics_dt()

        self.sim.set_camera_view(
            eye=[5.0, 0.0, 4.0],
            target=[0.0, 0.0, 1.0],
        )

        self.scene_cfg = NavisimWarehouseSceneCfg(
            num_envs=num_envs,
            env_spacing=4.0,
        )
        self.scene = InteractiveScene(self.scene_cfg)

        self.step_count = torch.zeros(num_envs, dtype=torch.long)
        self.prev_x = None

        self.sim.reset()
        self.reset()
    
    def set_up_scene(self, scene):
        # Load warehouse first
        sim_utils.create_prim(
            prim_path="/World/Warehouse",
            usd_path=WAREHOUSE_USD
        )

        # Let IsaacLab build all articulations & sensors
        scene.clone_environments()

    def reset(self):
        root = self.scene["jetbot"].data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins

        self.scene["jetbot"].write_root_pose_to_sim(root[:, :7])
        self.scene["jetbot"].write_root_velocity_to_sim(root[:, 7:])

        self.scene["jetbot"].write_joint_state_to_sim(
            self.scene["jetbot"].data.default_joint_pos,
            self.scene["jetbot"].data.default_joint_vel,
        )

        self.scene.reset()
        self.step_count[:] = 0
        self.prev_x = self._get_root_x()

        return self._get_obs()

    def step(self, actions):
        self.scene["jetbot"].set_joint_velocity_target(actions)
        self.scene.write_data_to_sim()

        self.sim.step()
        self.scene.update(self.sim_dt)
        self._maybe_save_frame()


        reward = self._compute_reward()
        done = self.step_count >= self.max_episode_len
        self.step_count += 1

        return self._get_obs(), reward, done, {}

    def _get_root_x(self):
        return self.scene["jetbot"].data.root_state_w[:, 0].clone()

    def _get_obs(self):
        return self.scene["jetbot"].data.root_state_w.clone()

    def _compute_reward(self):
        curr_x = self._get_root_x()
        delta = curr_x - self.prev_x
        self.prev_x = curr_x
        return delta

    def _maybe_save_frame(self):
        if self.frame % self.save_every != 0:
            self.frame += 1
            return

        rgb = self.scene["camera"].data.output["rgb"]  # shape [N, H, W, 4]
        img = (rgb[0, :, :, :3] * 255).astype(np.uint8)

        imageio.imwrite(self.frame_dir / f"frame_{self.frame:05d}.png", img)
        self.frame += 1
