"""Minimal GLB (glTF 2.0 binary) writer/reader - no third-party dependencies."""
import json, struct, io
import numpy as np
from PIL import Image

ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class GLBBuilder:
    def __init__(self):
        self.bin = bytearray(); self.views = []; self.accessors = []
        self.images = []; self.textures = []; self.materials = []

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b"\0"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data; self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, atype, target=None, minmax=False):
        ctype = {np.dtype("float32"): 5126, np.dtype("uint16"): 5123, np.dtype("uint32"): 5125}[arr.dtype]
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": atype}
        if minmax:
            a["min"] = arr.min(0).astype(float).tolist(); a["max"] = arr.max(0).astype(float).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1

    def image(self, pil, fmt):
        b = io.BytesIO()
        if fmt == "PNG":
            pil.save(b, "PNG", optimize=True)
        else:
            pil.save(b, "JPEG", quality=88, optimize=True)
        self.images.append({"bufferView": self._view(b.getvalue()),
                            "mimeType": "image/png" if fmt == "PNG" else "image/jpeg"})
        return len(self.images) - 1

    def texture(self, img_idx):
        self.textures.append({"sampler": 0, "source": img_idx})
        return len(self.textures) - 1

    def primitive(self, pos, nrm, idx, uv=None, col=None, tan=None, material=0):
        at = {"POSITION": self.accessor(pos, "VEC3", ARRAY_BUFFER, True),
              "NORMAL": self.accessor(nrm, "VEC3", ARRAY_BUFFER)}
        if tan is not None:
            at["TANGENT"] = self.accessor(tan, "VEC4", ARRAY_BUFFER)
        if uv is not None:
            at["TEXCOORD_0"] = self.accessor(uv, "VEC2", ARRAY_BUFFER)
        if col is not None:
            at["COLOR_0"] = self.accessor(col, "VEC3", ARRAY_BUFFER)
        return {"attributes": at, "indices": self.accessor(idx.reshape(-1), "SCALAR", ELEMENT_ARRAY_BUFFER),
                "material": material, "mode": 4}

    def save(self, path, prims, name):
        g = {"asset": {"version": "2.0", "generator": "ARMR-A1 pipeline/glbtools.py"},
             "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
             "meshes": [{"name": name, "primitives": prims}], "materials": self.materials,
             "accessors": self.accessors, "bufferViews": self.views,
             "buffers": [{"byteLength": len(self.bin)}]}
        if self.images:
            g["images"] = self.images; g["textures"] = self.textures
            g["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 33071}]
        js = json.dumps(g, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        bn = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(js) + 8 + len(bn)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(bn), 0x004E4942) + bn)


def read_glb(path):
    """Parse a GLB written by GLBBuilder -> primitives as numpy arrays + decoded images."""
    d = open(path, "rb").read()
    jl, _ = struct.unpack_from("<II", d, 12)
    g = json.loads(d[20:20 + jl])
    bl, _ = struct.unpack_from("<II", d, 20 + jl)
    bn = d[28 + jl:28 + jl + bl]
    comp = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}
    nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = g["accessors"][i]; v = g["bufferViews"][a["bufferView"]]
        n = nc[a["type"]]
        x = np.frombuffer(bn, comp[a["componentType"]], a["count"] * n, v["byteOffset"])
        return x.reshape(-1, n) if n > 1 else x

    prims = []
    for p in g["meshes"][0]["primitives"]:
        q = {k: acc(i) for k, i in p["attributes"].items()}
        q["indices"] = acc(p["indices"]).astype(np.int64)
        q["material"] = g["materials"][p["material"]]
        prims.append(q)
    imgs = []
    for im in g.get("images", []):
        v = g["bufferViews"][im["bufferView"]]
        imgs.append(Image.open(io.BytesIO(bn[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])).convert("RGB"))
    return {"prims": prims, "images": imgs, "json": g}
