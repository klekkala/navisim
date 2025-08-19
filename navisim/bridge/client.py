# navisim/bridge/navisim_bridge_client.py
import zmq
import numpy as np
from PIL import Image
import io

def _read_tiff(sock) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(sock.recv())))

class NavisimBridgeClient:
    def __init__(self, url="ipc:///tmp/navisim/bridge"):
        ctx = zmq.Context.instance()
        self._req = ctx.socket(zmq.REQ)
        self._req.connect(url)

    def step_render(self, action: dict, dt: float, camera: dict, sector_id: str, version: str):
        self._req.send_json({
            "kind": "step_render",
            "action": action, "dt": dt,
            "camera": camera,
            "sector_id": sector_id, "version": version
        })
        header = self._req.recv_json()
        if not header.get("ok", False):
            raise RuntimeError(header.get("error", "step_render failed"))

        color = _read_tiff(self._req)        # HxWx{3,4} uint8
        inv_depth = _read_tiff(self._req)    # HxW float32
        return header, color, inv_depth

    def control(self, kind: str, **kwargs):
        msg = {"kind": kind}; msg.update(kwargs)
        self._req.send_json(msg)
        return self._req.recv_json()