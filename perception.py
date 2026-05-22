import time

import numpy as np
from pyrecord3d.sale import Record3DStream, Record3DVideo


class Perception:
    """
    Builds a point cloud from Record3D and runs AnyGrasp on it.

    Two modes:
      - Live USB stream : Perception(anygrasp, T, mode="stream")
      - Saved .r3d file : Perception(anygrasp, T, mode="file", file_path="scene.r3d")
    """

    def __init__(
        self,
        anygrasp,
        camera_to_robot: np.ndarray,
        mode: str = "stream",
        file_path: str = None,
        frame_index: int = None,
        depth_min: float = 0.1,
        depth_max: float = 1.5,
        workspace_lims=None,
    ):
        """
        anygrasp        : initialized AnyGrasp instance
        camera_to_robot : 4x4 transform from camera frame to robot base frame
        mode            : "stream" for live USB, "file" for saved .r3d
        file_path       : path to .r3d file (required when mode="file")
        frame_index     : which frame to use from the file (None = best frame)
        depth_min/max   : clip range in metres
        workspace_lims  : [[xmin,xmax],[ymin,ymax],[zmin,zmax]] in camera frame
        """
        if mode not in ("stream", "file"):
            raise ValueError("mode must be 'stream' or 'file'")
        if mode == "file" and not file_path:
            raise ValueError("file_path is required when mode='file'")

        self.anygrasp = anygrasp
        self.T_cam2robot = camera_to_robot
        self.mode = mode
        self.frame_index = frame_index
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.lims = workspace_lims or [
            [-0.5, 0.5],
            [-0.5, 0.5],
            [depth_min, depth_max],
        ]

        if mode == "stream":
            self.session = Record3DStream()
            self.session.connect()
            print("[Perception] Connected to Record3D USB stream.")
        else:
            self.video = Record3DVideo(file_path)
            n = self.video.get_n_frames()
            print(f"[Perception] Loaded '{file_path}' — {n} frames.")

    # ------------------------------------------------------------------
    def _get_frame(self):
        """Return (depth, rgb, K) from either source."""
        if self.mode == "stream":
            return self._get_stream_frame()
        else:
            return self._get_file_frame()

    def _get_stream_frame(self):
        frame_ready = [False]

        @self.session.on_new_frame
        def _cb():
            frame_ready[0] = True

        while not frame_ready[0]:
            time.sleep(0.005)

        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        K = self.session.get_intrinsic_mat()
        return depth, rgb, K

    def _get_file_frame(self):
        n = self.video.get_n_frames()

        if self.frame_index is not None:
            idx = self.frame_index
        else:
            # Use the middle frame — avoids motion blur at start/end of recording
            idx = n // 2

        depth = self.video.get_depth_frame(idx)
        rgb = self.video.get_rgb_frame(idx)
        K = self.video.get_intrinsic_mat()
        print(f"[Perception] Using frame {idx}/{n}")
        return depth, rgb, K

    # ------------------------------------------------------------------
    def _depth_to_pointcloud(self, depth, rgb, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z = depth.astype(np.float32)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        mask = (z > self.depth_min) & (z < self.depth_max)
        points = np.stack([x, y, z], axis=-1)[mask]
        colors = (rgb[mask] / 255.0).astype(np.float32)
        return points, colors

    # ------------------------------------------------------------------
    def get_best_grasp(self):
        """
        Returns (grasp_pose_robot, width) or (None, None) if no grasp found.

        grasp_pose_robot : 4x4 pose in robot base frame
        width            : suggested gripper opening in metres
        """
        depth, rgb, K = self._get_frame()
        points, colors = self._depth_to_pointcloud(depth, rgb, K)

        if len(points) < 50:
            print("[Perception] Not enough points in scene.")
            return None, None

        gg, _ = self.anygrasp.get_grasp(points, colors, lims=self.lims)

        if gg is None or len(gg) == 0:
            print("[Perception] AnyGrasp found no grasps.")
            return None, None

        gg = gg.nms().sort_by_score()
        best = gg[0]

        pose_cam = np.eye(4)
        pose_cam[:3, :3] = best.rotation_matrix
        pose_cam[:3, 3] = best.translation

        pose_robot = self.T_cam2robot @ pose_cam

        print(
            f"[Perception] Best grasp  score={best.score:.3f}  "
            f"width={best.width:.4f}m  "
            f"pos={pose_robot[:3, 3]}"
        )
        return pose_robot, float(best.width)
