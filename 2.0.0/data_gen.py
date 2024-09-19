import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import gymnasium
import gymnasium.spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
import metaworld
import mujoco
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from common.pointcloud_util import pointcloud_subsampling
from typing import Optional, Tuple, Dict, Union, List

TASK_BOUNDS = {
    'default': [
        [-0.4, 0.3, -1e-3],     # lb
        [0.4, 0.9, 0.7]         # ub
    ],
}


class MetaworldEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array'], 
        'render_fps': 10,
    }

    def __init__(self,
        task_name: str,
        image_size: int = 128,
        num_points: int = 512,
        seed: Optional[int] = None,
        camera_names: List[str] = [
            'topview', 'corner', 'corner2', 
            'corner3', 'behindGripper', 'gripperPOV'
        ],
        oracle: bool = False,
        video_camera: str = 'corner2',
        max_episode_steps: int = 200
    ):
        super().__init__()
        # env seeding and instantiation
        task_name += '-v2-goal-observable'
        env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name](seed=seed)
        self.env = env
        if seed is not None:
            self.seed(seed)
        env._freeze_rand_vec = not oracle

        # adjust camera near-far
        env.model.vis.map.znear = 0.1
        env.model.vis.map.zfar = 1.5

        # NOTE: hack corner2 camera setup, https://arxiv.org/abs/2212.05698
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'corner2')
        assert cam_id == 2      # human knowledge
        env.model.cam_pos0[cam_id] = [0.6, 0.295, 0.8]
        env.model.cam_pos[cam_id] = [0.6, 0.295, 0.8]

        # setup camera properties
        camera_ids = {
            name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in camera_names
        }

        camera_renderer = {
            name: MujocoRenderer(
                env.model, env.data, None, 
                image_size, image_size, 1000, None, name, {}
            ) for name in camera_names
        }
        camera_o3d_pinhole = {
            name: o3d.camera.PinholeCameraIntrinsic(
                image_size, image_size, 
                image_size / (2 * np.tan(np.radians(env.model.cam_fovy[cam_id]) / 2)),
                image_size / (2 * np.tan(np.radians(env.model.cam_fovy[cam_id]) / 2)),
                image_size / 2, image_size / 2
            )
            for name, cam_id in camera_ids.items()
        }
        camera_poses = {name: np.eye(4) for name in camera_names}
        for cam_name, cam_id in camera_ids.items():
            camera_poses[cam_name][:3, 3] = env.model.cam_pos0[cam_id]
            camera_poses[cam_name][:3, :3] = np.matmul(
                env.model.cam_mat0[cam_id].reshape(3, 3), 
                R.from_quat([1, 0, 0, 0]).as_matrix()       # mujoco cam correction
            )
        
        # setup gym spaces
        observation_space = gymnasium.spaces.Dict()
        # rgb, depth
        for name in camera_names:
            observation_space[f"{name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, 
                shape=(3, image_size, image_size), 
                dtype=np.uint8
            )
            observation_space[f"{name}_depth"] = gymnasium.spaces.Box(
                low=0, high=255,
                shape=(image_size, image_size),
                dtype=np.float32
            )
        # pointcloud
        observation_space["fused_pointcloud"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_points, 3),
            dtype=np.float32
        )
        # robot state
        observation_space['agent_pos'] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.get_robot_state().shape,
            dtype=np.float32
        )
        # oracle
        if oracle:
            observation_space['full_state'] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=env.observation_space.shape,
                dtype=np.float32
            )

        # attr
        self.env = env
        self.camera_names = camera_names
        self.camera_ids = camera_ids
        self.camera_renderer = camera_renderer
        self.camera_o3d_pinhole = camera_o3d_pinhole
        self.camera_poses = camera_poses
        self.oracle = oracle
        self.observation_space = observation_space
        self.image_size = image_size
        self.num_points = num_points
        self.task_name = task_name
        self.task_bbox = np.array(TASK_BOUNDS.get(  # to crop pointcloud
            task_name, TASK_BOUNDS['default']))
        self.video_camera = video_camera
        self.episode_length = self._max_episode_steps = max_episode_steps
        self.action_space = env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]


    def get_robot_state(self) -> np.ndarray:
        return np.concatenate([
            self.env.get_endeff_pos(),
            self.env._get_site_pos('leftEndEffector'),
            self.env._get_site_pos('rightEndEffector')
        ])
    

    def get_rgb(self, cam_name: List = None) -> Dict[str, np.ndarray]:
        if cam_name is None:
            cam_name = self.camera_names
        return {
            cam: np.flip(self.camera_renderer[cam].render('rgb_array'), axis=0)
            for cam in cam_name
        }
    

    # https://github.com/openai/mujoco-py/issues/520
    def get_depth(self, cam_name: List = None) -> Dict[str, np.ndarray]:
        if cam_name is None:
            cam_name = self.camera_names

        # depth scaling params
        extent = self.env.model.stat.extent
        near = self.env.model.vis.map.znear * extent
        far = self.env.model.vis.map.zfar * extent

        return {
            cam: near / (1 - 
                np.flip(self.camera_renderer[cam].render('depth_array'), axis=0) * (1 - near / far))
            for cam in cam_name
        }
    

    def get_pointcloud(self, 
        cam_name: List = None, return_depth: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        # fuse if multiple cameras
        if cam_name is None:
            cam_name = self.camera_names

        depths = self.get_depth(cam_name)
        
        result_o3d_pointcloud = o3d.geometry.PointCloud()
        for cam in cam_name:
            result_o3d_pointcloud += o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(np.ascontiguousarray(np.flip(depths[cam], axis=0))),
                self.camera_o3d_pinhole[cam],
                np.linalg.inv(self.camera_poses[cam])
            )
        pointcloud = np.asarray(result_o3d_pointcloud.points)

        # crop
        pointcloud = pointcloud[np.all((
            pointcloud > self.task_bbox[0]) & 
            (pointcloud < self.task_bbox[1]), 
        axis=-1)]

        # subsample
        pointcloud = pointcloud_subsampling(pointcloud, self.num_points, method='fps')

        if return_depth:
            return pointcloud, depths
        else:
            return pointcloud
        

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        rgbs: Dict
        depths: Dict
        pointcloud: np.ndarray
        robot_state: np.ndarray
        rgbs = self.get_rgb()
        robot_state = self.get_robot_state()
        pointcloud, depths = self.get_pointcloud(return_depth=True)

        for k, v in rgbs.items():
            rgbs[k] = v.transpose(2, 0, 1)
        
        obs_dict = {}
        for cam in self.camera_names:
            obs_dict[f"{cam}_rgb"] = rgbs[cam]
            obs_dict[f"{cam}_depth"] = depths[cam]
        obs_dict[f"fused_pointcloud"] = pointcloud
        obs_dict['agent_pos'] = robot_state
        return obs_dict
    

    def step(self, action: np.ndarray):
        full_state, reward, done, truncate, info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self.get_obs_dict()
        if self.oracle:
            obs_dict['full_state'] = full_state
        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, truncate, info
    

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        self.env.reset()
        self.env.reset_model()
        full_state, info = self.env.reset()
        self.cur_step = 0
        obs_dict = self.get_obs_dict()
        if self.oracle:
            obs_dict['full_state'] = full_state
        return obs_dict, info
    

    def seed(self, seed=None):
        self.env.seed(seed)


    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        return self.get_rgb([self.video_camera,])[self.video_camera]
    

    def close(self):
        self.env.close()
        