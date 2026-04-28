from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "physics_prim_path": "/physicsScene"})

import omni.ui as ui
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import RigidPrim
from pxr import Sdf, UsdPhysics, Gf, UsdGeom
import omni.usd
import numpy as np
import h5py
import carb
import os
import time
import random
from typing import List, Optional

# ==============================================================================
# Path Configuration
# ==============================================================================
HDF5_ROLLOUT_FOLDER = "/yourpath/gpu_0/temp_hdf5_rollouts"
A2_USD_PATH        = "/yourpath/assets/robot/AgiBotA2/model_no_col2.usd"
SCENE_USD_PATH     = "/yourpath/Collected_kitchen/kitchen.usd"

# Prim paths in the stage
A2_PRIM_PATH    = "/World/A2"
SCENE_PRIM_PATH = "/scene"
OBJECT_PRIM_PATHS = [
    "/scene/new_leshi/Mesh0",
    "/scene/new_greentee/Mesh0",
    "/scene/new_yellowtea/Mesh0",
]

# HDF5 object name -> stage prim path mapping
HDF5_TO_PRIM_MAP = {
    "leshi":    "/scene/new_leshi/Mesh0",
    "greentee": "/scene/new_greentee/Mesh0",
    "yellowtea":"/scene/new_yellowtea/Mesh0",
}

# ==============================================================================
# Global State
# ==============================================================================
g_world:                          Optional[World]           = None
g_a2_robot:                       Optional[Robot]           = None
g_tracked_objects:                List[RigidPrim]           = []
g_a2_robot_view:                  Optional[ArticulationView]= None

g_rollout_files:                  List[str]                 = []   # all .hdf5 files in folder
g_hdf5_file:                      Optional[h5py.File]       = None # currently open file
g_current_rollout_path:           Optional[str]             = None # path of the open file
g_current_demo_key:               Optional[str]             = None # demo key inside the file

g_is_playing:                     bool                      = False
g_current_frame_index:            int                       = 0
g_total_frames:                   int                       = 0
g_actions_data:                   Optional[np.ndarray]      = None
g_full_replay_joint_indices_38dof:Optional[np.ndarray]      = None

RECORDING_FREQUENCY:              float                     = 30.0
g_replay_start_wall_time:         float                     = 0.0
g_keyboard_sub                                              = None

# ==============================================================================
# 26-DOF -> 38-DOF Mimic Expansion
# ==============================================================================
def expand_26dof_to_full_mimic_action(action_26d: np.ndarray) -> np.ndarray:
    """Expand a 26-dim action vector to 38-dim via mimic rules (14 arm + 24 hand)."""
    if action_26d.shape[0] != 26:
        carb.log_error_once(f"Expected 26-dim input, got {action_26d.shape[0]}.")
        return np.zeros(38, dtype=np.float32)

    action_38d = np.zeros(38, dtype=np.float32)

    # 14 arm joints: direct 1-to-1 copy
    action_38d[0:14] = action_26d[0:14]

    # Left hand: 6 master joints -> 12 mimic joints
    action_38d[14]    = action_26d[14]
    action_38d[15:18] = action_26d[15]
    action_38d[18:20] = action_26d[16]
    action_38d[20:22] = action_26d[17]
    action_38d[22:24] = action_26d[18]
    action_38d[24:26] = action_26d[19]

    # Right hand: 6 master joints -> 12 mimic joints
    action_38d[26]    = action_26d[20]
    action_38d[27:30] = action_26d[21]
    action_38d[30:32] = action_26d[22]
    action_38d[32:34] = action_26d[23]
    action_38d[34:36] = action_26d[24]
    action_38d[36:38] = action_26d[25]

    return action_38d


# ==============================================================================
# Utility
# ==============================================================================
def apply_translate_op(xform_prim, z_offset: float):
    """Set (or add) a translate XformOp with the given z offset."""
    if xform_prim and xform_prim.IsValid():
        xform = UsdGeom.Xformable(xform_prim)
        op = next((op for op in xform.GetOrderedXformOps()
                   if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if op:
            op.Set(Gf.Vec3d(0, 0, z_offset))
        else:
            xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, z_offset))
    else:
        carb.log_warn("Cannot apply translate op: prim is None or invalid.")


# ==============================================================================
# HDF5 File Management
# ==============================================================================
def scan_rollout_folder() -> bool:
    """Scan HDF5_ROLLOUT_FOLDER and populate g_rollout_files."""
    global g_rollout_files
    if not os.path.isdir(HDF5_ROLLOUT_FOLDER):
        carb.log_error(f"Rollout folder not found: {HDF5_ROLLOUT_FOLDER}")
        return False
    files = sorted([
        os.path.join(HDF5_ROLLOUT_FOLDER, f)
        for f in os.listdir(HDF5_ROLLOUT_FOLDER)
        if f.endswith(".hdf5") or f.endswith(".h5")
    ])
    if not files:
        carb.log_error(f"No .hdf5 files found in: {HDF5_ROLLOUT_FOLDER}")
        return False
    g_rollout_files = files
    print(f"Found {len(g_rollout_files)} rollout file(s) in folder.")
    return True


def open_rollout_file(path: str) -> bool:
    """Close any open HDF5 file, then open the specified rollout file."""
    global g_hdf5_file, g_current_rollout_path, g_current_demo_key
    if g_hdf5_file:
        g_hdf5_file.close()
        g_hdf5_file = None

    if not os.path.exists(path):
        carb.log_error(f"Rollout file not found: {path}")
        return False
    try:
        g_hdf5_file = h5py.File(path, 'r')
        g_current_rollout_path = path
        demo_keys = sorted(list(g_hdf5_file.get('data', {}).keys()))
        if not demo_keys:
            carb.log_error(f"No demo keys found in: {path}")
            g_hdf5_file.close(); g_hdf5_file = None
            return False
        g_current_demo_key = demo_keys[0]
        print(f"Opened rollout: {os.path.basename(path)}  demo_key={g_current_demo_key}")
        return True
    except Exception as e:
        carb.log_error(f"Failed to open HDF5 file {path}: {e}")
        return False


# ==============================================================================
# Scene Setup
# ==============================================================================
def setup_scene():
    global g_world, g_a2_robot, g_tracked_objects, g_a2_robot_view
    global g_full_replay_joint_indices_38dof

    g_world = World(stage_units_in_meters=1.0, physics_dt=1.0/120, rendering_dt=1.0/60.0)
    g_world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=A2_USD_PATH, prim_path=A2_PRIM_PATH)
    if os.path.exists(SCENE_USD_PATH):
        add_reference_to_stage(usd_path=SCENE_USD_PATH, prim_path=SCENE_PRIM_PATH)

    g_a2_robot      = Robot(prim_path=A2_PRIM_PATH, name="a2_robot_replay")
    g_a2_robot_view = ArticulationView(prim_paths_expr=A2_PRIM_PATH, name="a2_robot_view")
    g_world.scene.add(g_a2_robot)

    for _ in range(10):
        simulation_app.update()
    g_world.reset()
    g_a2_robot_view.initialize()

    # Register tracked rigid objects
    stage = omni.usd.get_context().get_stage()
    for i, prim_path in enumerate(OBJECT_PRIM_PATHS):
        if stage.GetPrimAtPath(prim_path):
            obj = RigidPrim(prim_path=prim_path, name=f"object_replay_{i}")
            if obj:
                g_world.scene.add(obj)
                obj.initialize()
                g_tracked_objects.append(obj)

    # Build 38-DOF joint index array
    full_joint_names_38dof = [
        'idx13_left_arm_joint1',  'idx14_left_arm_joint2',  'idx15_left_arm_joint3',
        'idx16_left_arm_joint4',  'idx17_left_arm_joint5',  'idx18_left_arm_joint6',
        'idx19_left_arm_joint7',  'idx20_right_arm_joint1', 'idx21_right_arm_joint2',
        'idx22_right_arm_joint3', 'idx23_right_arm_joint4', 'idx24_right_arm_joint5',
        'idx25_right_arm_joint6', 'idx26_right_arm_joint7',
        'L_thumb_swing_joint', 'L_thumb_1_joint', 'L_thumb_2_joint', 'L_thumb_3_joint',
        'L_index_1_joint',  'L_index_2_joint',  'L_middle_1_joint', 'L_middle_2_joint',
        'L_ring_1_joint',   'L_ring_2_joint',   'L_little_1_joint', 'L_little_2_joint',
        'R_thumb_swing_joint', 'R_thumb_1_joint', 'R_thumb_2_joint', 'R_thumb_3_joint',
        'R_index_1_joint',  'R_index_2_joint',  'R_middle_1_joint', 'R_middle_2_joint',
        'R_ring_1_joint',   'R_ring_2_joint',   'R_little_1_joint', 'R_little_2_joint',
    ]

    try:
        indices = [g_a2_robot.get_dof_index(n) for n in full_joint_names_38dof]
        missing = [n for n, idx in zip(full_joint_names_38dof, indices) if idx == -1]
        if missing:
            carb.log_error(f"Joint(s) not found in robot model: {missing}"); return
        g_full_replay_joint_indices_38dof = np.array(indices, dtype=np.int32)
        print(f"Mapped {len(g_full_replay_joint_indices_38dof)} replay joints (incl. mimic).")
    except Exception as e:
        carb.log_error(f"Error mapping joint indices: {e}"); return

    # Apply high gains for finger joints
    finger_names = {j for j in full_joint_names_38dof if ('L_' in j or 'R_' in j)}
    FINGER_KP, FINGER_KD = 1.0e10, 6.00e7
    DEFAULT_KP, DEFAULT_KD = 400.0, 40.0
    if g_a2_robot_view and g_a2_robot_view.is_valid():
        num_dofs = g_a2_robot_view.num_dof
        kps = np.full(num_dofs, DEFAULT_KP)
        kds = np.full(num_dofs, DEFAULT_KD)
        for i, name in enumerate(g_a2_robot_view.dof_names):
            if name in finger_names:
                kps[i], kds[i] = FINGER_KP, FINGER_KD
        g_a2_robot_view.set_gains(kps, kds)
        print("Finger joint gains updated.")

    # Fix robot base to ground via a fixed joint
    ground_prim = g_world.scene.get_object("default_ground_plane").prim
    torso_prim  = stage.GetPrimAtPath(f"{A2_PRIM_PATH}/raise_a2_t2d0_flagship/base_link")
    if ground_prim.IsValid() and torso_prim.IsValid():
        joint_path = Sdf.Path("/World/ground_torso_fixed_joint_A2_replay")
        if not stage.GetPrimAtPath(joint_path):
            joint = UsdPhysics.FixedJoint.Define(g_world.stage, joint_path)
            joint.CreateBody0Rel().SetTargets([ground_prim.GetPath()])
            joint.CreateBody1Rel().SetTargets([torso_prim.GetPath()])

    # Offset ground plane visuals to match robot foot height
    z_bias = -1.20
    apply_translate_op(stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane"), z_bias)
    apply_translate_op(stage.GetPrimAtPath("/World/defaultGroundPlane/Environment"), z_bias)
    print("Scene setup complete.")


# ==============================================================================
# Demo Load & Reset
# ==============================================================================
def reset_and_load_demo(demo_key: str):
    """Reset the scene to the initial state of a demo and load its action sequence."""
    global g_is_playing, g_current_frame_index, g_total_frames, g_actions_data
    g_is_playing = False

    if not g_hdf5_file or 'data' not in g_hdf5_file or demo_key not in g_hdf5_file['data']:
        carb.log_error(f"Cannot load demo: {demo_key}"); return

    print(f"Resetting scene to initial state of {demo_key} ...")
    data_group    = g_hdf5_file['data'][demo_key]
    init_state    = data_group['initial_state']

    # --- Robot initial joint positions ---
    if 'articulation/robot/joint_position' in init_state and g_full_replay_joint_indices_38dof is not None:
        init_pos_26d = init_state['articulation/robot/joint_position'][0]
        if init_pos_26d.shape[0] == 26:
            init_pos_38d = expand_26dof_to_full_mimic_action(init_pos_26d)
            full_pos     = g_a2_robot_view.get_joint_positions(clone=True)[0]
            full_pos[g_full_replay_joint_indices_38dof] = init_pos_38d
            g_a2_robot.set_joint_positions(full_pos)
            g_a2_robot.set_joint_velocities(np.zeros_like(full_pos))
        else:
            carb.log_error(f"Initial joint position dim={init_pos_26d.shape[0]}, expected 26.")
    else:
        carb.log_error("Initial joint position data or joint indices not available.")

    # --- Object initial poses ---
    prim_to_obj = {obj.prim_path: obj for obj in g_tracked_objects}
    for hdf5_name, prim_path in HDF5_TO_PRIM_MAP.items():
        hdf5_pose_path = f'rigid_object/{hdf5_name}/root_pose'
        if hdf5_pose_path not in init_state:
            carb.log_warn(f"No pose data for '{hdf5_name}' in HDF5 (path: {hdf5_pose_path}).")
            continue
        obj_pose    = init_state[hdf5_pose_path][0]
        pos         = obj_pose[:3]
        quat_xyzw   = obj_pose[3:]
        quat_wxyz   = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        if prim_path in prim_to_obj:
            tracked_obj = prim_to_obj[prim_path]
            tracked_obj.set_world_pose(position=pos, orientation=quat_wxyz)
            tracked_obj.set_linear_velocity(np.zeros(3))
            tracked_obj.set_angular_velocity(np.zeros(3))
        else:
            carb.log_warn(f"Prim '{prim_path}' not found in tracked objects.")

    # --- Load action sequence ---
    if 'actions' in data_group:
        g_actions_data = data_group['actions'][:]
        g_total_frames = g_actions_data.shape[0]
        g_current_frame_index = 0
        if g_actions_data.shape[1] != 26:
            carb.log_error(f"Action dim={g_actions_data.shape[1]}, expected 26.")
            g_actions_data = None; g_total_frames = 0
    else:
        g_actions_data = None; g_total_frames = 0
        carb.log_error(f"No 'actions' dataset found in demo '{demo_key}'.")

    # Let physics settle
    for _ in range(5):
        g_world.step(render=False)
    print("Scene reset complete. Press 'S' to start playback.")


# ==============================================================================
# Playback Controls
# ==============================================================================
def toggle_playback():
    """Toggle play / pause. Auto-resets if playback has finished."""
    global g_is_playing, g_replay_start_wall_time, g_current_frame_index
    if not g_current_demo_key:
        return
    if not g_is_playing and g_current_frame_index >= g_total_frames - 1:
        reset_current_playback()
    g_is_playing = not g_is_playing
    if g_is_playing:
        g_replay_start_wall_time = time.time() - (g_current_frame_index / RECORDING_FREQUENCY)
        print("--- REPLAY PLAYING ---")
    else:
        print("--- REPLAY PAUSED ---")


def reset_current_playback():
    """Reload the current demo from the beginning."""
    if g_current_demo_key:
        print(f"--- RESETTING demo {g_current_demo_key} ---")
        reset_and_load_demo(g_current_demo_key)


def select_random_rollout():
    """Randomly pick a rollout file from the folder and start playback from its first demo."""
    global g_current_demo_key
    if not g_rollout_files:
        carb.log_warn("No rollout files available.")
        return

    chosen_path = random.choice(g_rollout_files)
    print(f"--- RANDOM ROLLOUT: {os.path.basename(chosen_path)} ---")
    if open_rollout_file(chosen_path):
        reset_and_load_demo(g_current_demo_key)


def keyboard_event_cb(event, *args, **kwargs):
    """Handle keyboard input: S=play/pause, R=reset, N=random rollout."""
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if   event.input == carb.input.KeyboardInput.S: toggle_playback()
        elif event.input == carb.input.KeyboardInput.R: reset_current_playback()
        elif event.input == carb.input.KeyboardInput.N: select_random_rollout()
    return True


# ==============================================================================
# Main Entry
# ==============================================================================
def main():
    global g_current_demo_key, g_keyboard_sub, g_is_playing, g_actions_data
    global g_full_replay_joint_indices_38dof, g_replay_start_wall_time
    global g_current_frame_index, g_total_frames

    setup_scene()

    if not scan_rollout_folder():
        print("Failed to scan rollout folder. Exiting.")
        simulation_app.close(); return

    # Open a random rollout on startup
    initial_path = random.choice(g_rollout_files)
    if not open_rollout_file(initial_path):
        print("Failed to open initial rollout file. Exiting.")
        simulation_app.close(); return

    app_input    = carb.input.acquire_input_interface()
    g_keyboard_sub = app_input.subscribe_to_keyboard_events(None, keyboard_event_cb)

    reset_and_load_demo(g_current_demo_key)

    print("\n" + "=" * 50)
    print("REPLAY CONTROLS:")
    print("  S : Play / Pause")
    print("  R : Reset current rollout")
    print("  N : Load a random rollout from folder")
    print("=" * 50 + "\n")

    g_world.play()

    while simulation_app.is_running():
        g_world.step(render=True)

        if g_is_playing and g_actions_data is not None and g_full_replay_joint_indices_38dof is not None:
            elapsed       = time.time() - g_replay_start_wall_time
            target_frame  = int(elapsed * RECORDING_FREQUENCY)

            if target_frame > g_current_frame_index:
                g_current_frame_index = target_frame

                if g_current_frame_index >= g_total_frames:
                    g_is_playing          = False
                    g_current_frame_index = g_total_frames - 1
                    print(f"--- REPLAY FINISHED: {os.path.basename(g_current_rollout_path)} ---")
                else:
                    action_26d = g_actions_data[g_current_frame_index]
                    action_38d = expand_26dof_to_full_mimic_action(action_26d)
                    g_a2_robot.get_articulation_controller().apply_action(
                        ArticulationAction(
                            joint_positions=action_38d,
                            joint_indices=g_full_replay_joint_indices_38dof,
                        )
                    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if g_keyboard_sub and hasattr(g_keyboard_sub, 'unsubscribe'):
            g_keyboard_sub.unsubscribe()
        if g_hdf5_file:
            g_hdf5_file.close()
            print("HDF5 file closed.")
        if 'simulation_app' in globals() and simulation_app.is_running():
            simulation_app.close()
