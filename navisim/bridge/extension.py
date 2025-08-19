# omni/navisim_bridge/extension.py
import io, json, time, numpy as np
import omni.ext, omni.usd, omni.kit.app, omni.timeline
import omni.physx as physx
import carb, zmq
from pxr import UsdGeom, Gf
from PIL import Image

# --------- helper utils ----------
def gf_mat4_to_np(m: Gf.Matrix4d) -> np.ndarray:
    return np.array(m, dtype=np.float64)

def prim_world_matrix(stage, path) -> np.ndarray:
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    m, _ = xform.GetLocalTransformation()
    return gf_mat4_to_np(m)

def np_to_tiff_bytes(arr: np.ndarray, float32=False) -> bytes:
    if float32:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
    else:
        a = arr
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype == np.float32 else a.astype(np.uint8)
        if a.ndim == 2:   # gray
            img = Image.fromarray(a)
        else:
            img = Image.fromarray(a)
    buf = io.BytesIO(); img.save(buf, format="TIFF")
    return buf.getvalue()

def tiff_bytes_to_np(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)))

# --------- external 3DGS render client ----------
class GSRenderClient:
    def __init__(self, url="ipc:///tmp/omni-3dgs-extension/vanillags_renderer"):
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.connect(url)

    def render(self, payload_json: dict, bg_rgba: np.ndarray, bg_depth: np.ndarray):
        # send JSON + bg rgba tiff + bg depth tiff
        self._sock.send_json(payload_json, flags=zmq.SNDMORE)
        self._sock.send(np_to_tiff_bytes(bg_rgba), flags=zmq.SNDMORE)
        self._sock.send(np_to_tiff_bytes(bg_depth, float32=True))
        meta = self._sock.recv_json()
        color = tiff_bytes_to_np(self._sock.recv())
        inv_depth = tiff_bytes_to_np(self._sock.recv())
        return meta, color, inv_depth

# --------- main extension ----------
class NavisimBridgeExt(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._stage = omni.usd.get_context().get_stage()
        self._app = omni.kit.app.get_app()
        self._timeline = omni.timeline.get_timeline_interface()
        self._physx_rb = physx.get_physx_rigid_body_interface()
        self._agent_path = "/World/Agent"   # adjust if needed
        self._agent_api = physx.get_physx_rigid_body_api(self._stage.GetPrimAtPath(self._agent_path))

        # Ensure simulation is running
        if not self._timeline.is_playing(): self._timeline.play()

        # ZMQ REP (Gym <-> Omni)
        ctx = zmq.Context.instance()
        self._rep = ctx.socket(zmq.REP)
        self._rep.bind("ipc:///tmp/navisim/bridge")

        # External 3DGS renderer client
        self._gs = GSRenderClient()

        # simple viewport grab (optional: Replicator for bg rgba/depth)
        self._width, self._height = 1280, 720

        # Poll each frame (non-blocking)
        self._sub = self._app.get_update_event_stream().create_subscription_to_pop(self._on_update)

    def on_shutdown(self):
        try: self._rep.close(0)
        except: pass
        self._sub = None

    # ---------- physics helpers ----------
    def _apply_action(self, a: dict):
        mode = a.get("mode", "vel")
        if mode == "vel":
            v = a.get("linear", [0,0,0]); w = a.get("angular",[0,0,0])
            self._agent_api.SetLinearVelocity(Gf.Vec3f(*v), True)
            self._agent_api.SetAngularVelocity(Gf.Vec3f(*w), True)
        elif mode == "force":
            f = a.get("force", [0,0,0]); t = a.get("torque",[0,0,0])
            self._agent_api.AddForce(Gf.Vec3f(*f), Gf.Vec3f(0,0,0))
            self._agent_api.AddTorque(Gf.Vec3f(*t))

    def _advance(self, dt: float):
        # advance wall-clock frames until we pass target time
        start = self._timeline.get_current_time()
        target = start + dt
        while self._timeline.get_current_time() + 1e-6 < target:
            self._app.update()

    def _state_out(self):
        T = prim_world_matrix(self._stage, self._agent_path)
        v = self._agent_api.GetLinearVelocity()
        w = self._agent_api.GetAngularVelocity()
        return {
            "T_world_agent": T.tolist(),
            "linear": [float(v[0]), float(v[1]), float(v[2])],
            "angular": [float(w[0]), float(w[1]), float(w[2])]
        }

    # ---------- camera + background ----------
    def _active_camera_msg(self, cam_req: dict) -> dict:
        # Use the agent transform as camera (agent-mounted camera). Adjust if using a separate camera prim.
        T_world_cam = prim_world_matrix(self._stage, self._agent_path)
        return {
            "T_world_cam": T_world_cam.tolist(),
            "width": int(cam_req.get("width", self._width)),
            "height": int(cam_req.get("height", self._height)),
            "fx": float(cam_req.get("fx", 900.0)),
            "fy": float(cam_req.get("fy", 900.0)),
            "cx": float(cam_req.get("cx", cam_req.get("width", self._width)/2.0)),
            "cy": float(cam_req.get("cy", cam_req.get("height", self._height)/2.0)),
            "convention": "omni", "near": 0.1, "far": 200.0
        }

    def _background_buffers(self, W: int, H: int):
        # Placeholder: provide blank BG; replace with Replicator RGBA/Depth capture if desired
        rgba = np.zeros((H, W, 3), np.uint8)     # or 4-channel if you like
        depth = np.full((H, W), np.inf, np.float32)
        return rgba, depth

    # ---------- main handler ----------
    def _handle_step_render(self, msg: dict):
        action = msg.get("action", {})
        dt = float(msg.get("dt", 1.0/60.0))
        sector_id = msg.get("sector_id", "")
        version = msg.get("version", "v0")
        cam = self._active_camera_msg(msg.get("camera", {}))

        # 1) physics
        self._apply_action(action)
        self._advance(dt)
        state = self._state_out()

        # 2) background
        W, H = cam["width"], cam["height"]
        bg_rgba, bg_depth = self._background_buffers(W, H)

        # 3) 3DGS render
        payload = {
            "command": "render",
            "sector_id": sector_id, "version": version,
            "camera": cam
        }
        t0 = time.time()
        meta, color, inv_depth = self._gs.render(payload, bg_rgba, bg_depth)
        render_ms = (time.time() - t0) * 1000.0

        # 4) reply: JSON + color TIFF + inv-depth TIFF
        header = {
            "ok": True,
            "state": state,
            "render_ms": render_ms,
            "sector_id": sector_id,
            "version": version
        }
        self._rep.send_json(header, flags=zmq.SNDMORE)
        self._rep.send(np_to_tiff_bytes(color), flags=zmq.SNDMORE)
        self._rep.send(np_to_tiff_bytes(inv_depth, float32=True))

    def _handle_control(self, msg: dict):
        kind = msg["kind"]
        # Forward control to the 3DGS server if you like (requires adding a control() RPC there),
        # or handle Omniverse-side visuals/camera/prim toggles here.
        # For now just ack.
        self._rep.send_json({"ok": True, "handled": kind})

    def _on_update(self, _e):
        try:
            if self._rep.poll(0):
                msg = self._rep.recv_json(flags=zmq.NOBLOCK)
                kind = msg.get("kind")
                if kind == "step_render":
                    self._handle_step_render(msg)
                elif kind in ("load_sector", "activate_sector", "unload_sector", "update_points", "set_sector"):
                    self._handle_control(msg)
                else:
                    self._rep.send_json({"ok": False, "error": f"unknown kind {kind}"})
        except zmq.Again:
            pass
        except Exception as e:
            carb.log_error(f"NavisimBridge error: {e}")
            try: self._rep.send_json({"ok": False, "error": str(e)})
            except: pass