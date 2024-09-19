if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import argparse

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import zarr
import numpy as np
import copy

from common.input_util import wait_user_input
from .env import MetaworldEnv
from metaworld.policies import *


def load_expert_policy(task_name):
    if task_name == 'peg-insert-side':
        agent = 'SawyerPegInsertionSideV2Policy'
    else:
        agent = f"Sawyer{''.join([s.capitalize() for s in task_name.split('-')])}V2Policy"
    return eval(agent)()


def gen_metaworld_data(args):
    task_name = args.task_name
    num_episodes = args.num_episodes
    save_dir = args.save_dir
    sensors = args.sensors
    if save_dir is None:
        save_dir = os.path.join(args.root_data_dir, f'metaworld_{task_name}_expert_{num_episodes}.zarr')
    if sensors is None:
        sensors = ['topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV']
    else:
        sensors = sensors.split(',')

    if os.path.exists(save_dir):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_dir} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_dir}")
    os.mkdir(save_dir)    

    env = MetaworldEnv(
        task_name, 
        num_points=512,
        image_size=128,
        camera_names=sensors,
        device='cuda:0',
        oracle=True
    )
    expert = load_expert_policy(task_name)

    total_count = 0
    data_collected = {
        **{f"{sensor}_rgb": [] for sensor in sensors},
        **{f"{sensor}_depth": [] for sensor in sensors},
        'fused_pointcloud': [],
        'agent_pos': [],
        'action': [],
        'episode_ends': []       # index of the last step of the episode
    }
    data_dtype = {
        **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
        **{f"{sensor}_depth": 'float32' for sensor in sensors},
        'fused_pointcloud': 'float32',
        'agent_pos': 'float32',
        'action': 'float32',
        'episode_ends': 'int64'
    }

    episode_idx = 0
    while episode_idx < num_episodes:
        raw_state = env.reset()['full_state']
        obs_dict = env.get_obs_dict()
        done = False
        episode_reward = 0.
        episode_success = False
        episode_suceess_count = 0

        this_totoal_count = 0
        this_data_collected = {
            **{f"{sensor}_rgb": [] for sensor in sensors},
            **{f"{sensor}_depth": [] for sensor in sensors},
            'fused_pointcloud': [],
            'agent_pos': [],
            'action': [],
        }

        while not done:
            action = expert.get_action(raw_state)

            this_totoal_count += 1
            for key in this_data_collected.keys():
                if key != 'action':
                    this_data_collected[key].append(obs_dict[key])
            this_data_collected['action'].append(action)

            obs_dict, reward, done, info = env.step(action)
            raw_state = obs_dict['full_state']
            episode_reward += reward
            episode_success = episode_success or info['success']
            episode_suceess_count += info['success']

    
        if not episode_success or episode_suceess_count < 5:
            print(
                f"Episode {episode_idx} failed with "
                f"reward={episode_reward}, success_count={episode_suceess_count}, success={episode_success}"
            )
        else:
            episode_idx += 1
            total_count += this_totoal_count
            data_collected['episode_ends'].append(copy.deepcopy(total_count))
            for key in this_data_collected.keys():
                data_collected[key].extend(copy.deepcopy(this_data_collected[key]))
            print(f"Episode {episode_idx}, reward: {episode_reward}, success count: {episode_suceess_count}")

    # post-process data
    for key in data_collected.keys():
        if key == 'episode_ends':
            data_collected[key] = np.array(data_collected[key])
        else:
            data_collected[key] = np.stack(data_collected[key], axis=0)

    # report
    print('-' * 50)
    for k, v in data_collected.items():
        if k != 'episode_ends':
            print(f"{k}: {v.shape}, range: {v.min()}, {v.max()}")

    # save data collected
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    chunk_size = 100
    for k, v in data_collected.items():
        if k != 'episode_ends':
            zarr_data.create_dataset(
                k, data=v, 
                chunks=(chunk_size, *v.shape[1:]), 
                dtype=data_dtype[k],
                overwrite=True,
                compressor=compressor
            )
        else:
            zarr_meta.create_dataset(
                k, data=v,
                dtype=data_dtype[k],
                overwrite=True,
                compressor=compressor
            )

    # clean
    del data_collected
    del zarr_root, zarr_data, zarr_meta
    del env, expert


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', type=str, required=True)
    parser.add_argument('-d', '--root_data_dir', type=str, default='data')
    parser.add_argument('-s', '--save_dir', type=str, default=None)
    parser.add_argument('-c', '--num_episodes', type=int, default=10)
    parser.add_argument('-o', '--sensors', type=str, default=None)
    args = parser.parse_args()
    gen_metaworld_data(args)