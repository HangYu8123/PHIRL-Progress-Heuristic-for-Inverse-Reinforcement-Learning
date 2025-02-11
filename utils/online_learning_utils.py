import os
import json
import h5py
import numpy as np
import robosuite as suite
from imitation.data import types


def flatten_obs(obs, obs_keys):
    """
    Given an observation dictionary and a list of keys, extract the values,
    flatten them and concatenate into a single 1D numpy array.
    (Used only for action prediction.)
    """
    values = []
    for key in obs_keys:
        if key in obs:
            val = np.array(obs[key]).flatten()
            values.append(val)
    if values:
        return np.concatenate(values)
    else:
        return np.array([])


def convert_obs_list_to_structured(obs_list):
    """
    Convert a list of observation dictionaries into a structured NumPy array.
    This is needed because the reference code expects a single dataset with
    named fields (e.g. 'cube_pos').

    Assumes all observations have the same keys and that each key’s value is
    convertible to a NumPy array (e.g. a float vector).
    """
    if not obs_list:
        return np.array([])
    # Use the keys from the first observation.
    keys = list(obs_list[0].keys())
    dtype = []
    for key in keys:
        # Get the shape from the first observation’s value.
        val = np.array(obs_list[0][key])
        # We use float32 for compatibility.
        dtype.append((key, np.float32, val.shape))
    structured_array = np.empty(len(obs_list), dtype=dtype)
    for i, obs in enumerate(obs_list):
        for key in keys:
            structured_array[i][key] = np.array(obs[key], dtype=np.float32)
    return structured_array


def generate_demo(agent, env, obs_keys):
    """
    Run one rollout using the given agent and environment.

    The observation used for action prediction is preprocessed (flattened according
    to obs_keys), but the raw observation (a dict) is recorded for later replay.
    Also records the simulator state at each step.

    Returns:
        - trajectory_type: an instance of imitation.data.types.Trajectory.
        - traj: a dict with keys:
            'obs': a structured NumPy array (one per time step),
            'actions': a NumPy array of actions,
            'dones': a NumPy array of terminal flags,
            'states': a NumPy array of simulator states.
    """
    traj = {"obs": [], "actions": [], "dones": [], "states": []}

    # Reset the environment and record the initial observation and state.
    obs = env.reset()
    state = env.sim.get_state().flatten()
    traj["obs"].append(obs)
    traj["states"].append(state)

    # Use the filtered (flattened) observation for the agent.
    flat_obs = flatten_obs(obs, obs_keys)
    done = False

    while not done:
        action, _ = agent.predict(flat_obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        traj["actions"].append(action)
        traj["obs"].append(obs)
        traj["dones"].append(done)
        state = env.sim.get_state().flatten()
        traj["states"].append(state)
        flat_obs = flatten_obs(obs, obs_keys)

    # Convert lists to NumPy arrays (actions, dones, states) and convert the list
    # of observation dictionaries into a structured array.
    traj["actions"] = np.array(traj["actions"])
    traj["dones"] = np.array(traj["dones"])
    traj["states"] = np.array(traj["states"])
    traj["obs"] = convert_obs_list_to_structured(traj["obs"])

    trajectory_type = types.Trajectory(
        obs=traj["obs"],
        acts=traj["actions"],
        terminal=traj["dones"],
        infos=None,
    )
    return trajectory_type, traj


def save_demo_to_hdf5(traj, file_path, demo_key, env_args):
    """
    Save a generated demo (trajectory) into an HDF5 file in robosuite dataset format.
    
    The file structure will be as follows:
        /data  (group)
            env_args (attribute): JSON string containing a dict with keys
                                    'env_name' and 'env_kwargs'.
            /demo_key (group)  -- for example, "demo_1"
                obs      (dataset): structured array of observations.
                actions  (dataset): array of actions.
                dones    (dataset): array of terminal flags.
                states   (dataset): array of simulator states.
    
    Parameters:
        traj (dict): Trajectory dictionary containing keys 'obs', 'actions', 'dones', and 'states'.
        file_path (str): Path to the HDF5 file.
        demo_key (str): Key to identify the demo (e.g., 'demo_1').
        env_args (dict): Dictionary with environment arguments. It should have keys
                         'env_name' (string) and 'env_kwargs' (dict of keyword arguments).
    """
    # Open (or create) the file.
    if os.path.exists(file_path):
        hf = h5py.File(file_path, "a")
    else:
        hf = h5py.File(file_path, "w")
        data_grp = hf.create_group("data")
        data_grp.attrs["env_args"] = json.dumps(env_args)
    data_grp = hf["data"]
    if demo_key in data_grp:
        print(f"Demo key {demo_key} already exists; overwriting it.")
        del data_grp[demo_key]
    demo_grp = data_grp.create_group(demo_key)
    demo_grp.create_dataset("obs", data=traj["obs"])
    demo_grp.create_dataset("actions", data=traj["actions"])
    demo_grp.create_dataset("dones", data=traj["dones"])
    demo_grp.create_dataset("states", data=traj["states"])
    hf.close()


def annotate_demo(file_path, demo_key, env, collect_progress_times, key_obs):
    """
    Replay the demo specified by demo_key from the given HDF5 file, pausing at several
    indices to ask the user to input a progress value. The progress annotations are saved
    as a JSON file and returned.

    Parameters:
        file_path (str): Path to the HDF5 file containing the demo.
        demo_key (str): Key of the demo to annotate.
        collect_progress_times (int): Number of times to collect progress annotations.
        key_obs (str): Key in the observation (e.g., 'cube_pos') to display for context.
                     (It is assumed that this field exists in the structured observation.)
                     
    Returns:
        progress_data (list): A list of annotation segments, each a dict with:
            'start_step', 'end_step', 'start_progress', 'end_progress'.
    """
    # Open the file and load the demo.
    with h5py.File(file_path, "r") as hf:
        data_grp = hf["data"]
        if demo_key not in data_grp:
            print(f"Demo key {demo_key} not found in {file_path}.")
            return None
        demo_grp = data_grp[demo_key]
        actions = np.array(demo_grp["actions"])
        states = np.array(demo_grp["states"])
        # Also load the environment arguments.
        env_args_data = json.loads(hf["data"].attrs["env_args"])

    # Create the environment for replay.
    # env = suite.make(
    #     env_args_data["env_name"],
    #     **env_args_data["env_kwargs"],
    # )

    # Reset the environment and set its simulator state.
    initial_state = states[0]
    print("Initial state:", initial_state)
    env.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    #env.render()

    total_steps = len(actions)
    # Compute pause indices (evenly spaced; always include the final step)
    pause_indices = np.linspace(0, total_steps, collect_progress_times + 2, dtype=int)[1:-1]
    pause_indices = list(pause_indices)
    if pause_indices[-1] != total_steps - 1:
        pause_indices.append(total_steps - 1)

    progress_data = []
    print(f"Replaying demo {demo_key} (total steps: {total_steps}).")
    for i in range(total_steps):
        action = actions[i]
        obs, rwd, done, _ = env.step(action)
        env.render()
        if i in pause_indices:
            print(f"--- At step {i} ---")
            # Use .get() in case the key is not present.
            print("Object position:", obs[key_obs[0]])
            user_input = input("Please input the progress value at this step: ")
            # Validate input.
            while True:
                try:
                    progress_value = float(user_input)
                    break
                except ValueError:
                    user_input = input("Invalid input. Please input a numeric progress value: ")
            # Determine the start step based on pause indices.
            idx = pause_indices.index(i)
            start_step = int(pause_indices[idx - 1]) if idx > 0 else 0
            segment = {
                "start_step": start_step,
                "end_step": int(i),
                "start_progress": progress_data[-1]["end_progress"] if progress_data else 0.0,
                "end_progress": progress_value,
            }
            progress_data.append(segment)
        if done:
            break

    # # Save the progress annotation to a JSON file.
    # ann_dir = "progress_annotations"
    # os.makedirs(ann_dir, exist_ok=True)
    # annotation_file = os.path.join(ann_dir, f"{demo_key}.json")
    # with open(annotation_file, "w") as f_ann:
    #     json.dump(progress_data, f_ann, indent=4)
    # print(f"Annotation saved to {annotation_file}")
    # return progress_data
