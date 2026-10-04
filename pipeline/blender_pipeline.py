"""
Blender (bpy) version of the Part B workflow.  UNTESTED in the authoring environment - treat as a starting
point and adjust to your Blender version (written against the Blender 4.x Python API).

Usage:  blender -b --python blender_pipeline.py -- ../assets/vase_high_poly.glb ../web/assets/vase_blender_draco.glb
"""
import bpy, sys
src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
RATIO, TEX = 0.0083, 1024                                   # 6144 / 737280 triangles

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
hi = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in hi: o.select_set(True)
bpy.context.view_layer.objects.active = hi[0]; bpy.ops.object.join(); hi = bpy.context.active_object; hi.name = "high"

# 1) duplicate -> low poly, apply Decimate (Collapse)
bpy.ops.object.duplicate(); lo = bpy.context.active_object; lo.name = "low"
m = lo.modifiers.new("dec", "DECIMATE"); m.decimate_type = "COLLAPSE"; m.ratio = RATIO
bpy.ops.object.modifier_apply(modifier=m.name)

# 2) UV unwrap the low poly
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003); bpy.ops.object.mode_set(mode="OBJECT")

# 3) bake normal + diffuse (selected-to-active)
bpy.context.scene.render.engine = "CYCLES"
mat = bpy.data.materials.new("baked"); mat.use_nodes = True; lo.data.materials.clear(); lo.data.materials.append(mat)
nt = mat.node_tree
def img_node(name, color_space):
    im = bpy.data.images.new(name, TEX, TEX); im.colorspace_settings.name = color_space
    n = nt.nodes.new("ShaderNodeTexImage"); n.image = im; return n
n_col, n_nrm = img_node("diffuse", "sRGB"), img_node("normal", "Non-Color")
bpy.ops.object.select_all(action="DESELECT"); hi.select_set(True); lo.select_set(True); bpy.context.view_layer.objects.active = lo
bk = bpy.context.scene.render.bake; bk.use_selected_to_active = True; bk.cage_extrusion = 0.05
nt.nodes.active = n_nrm; bpy.ops.object.bake(type="NORMAL")
bk.use_pass_direct = bk.use_pass_indirect = False
nt.nodes.active = n_col; bpy.ops.object.bake(type="DIFFUSE")
bsdf = nt.nodes["Principled BSDF"]
nt.links.new(n_col.outputs["Color"], bsdf.inputs["Base Color"])
nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(n_nrm.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

# 4) export only the low poly with Draco
bpy.ops.object.select_all(action="DESELECT"); lo.select_set(True)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True,
                          export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=6,
                          export_draco_position_quantization=14, export_draco_normal_quantization=10, export_draco_texcoord_quantization=12)
print("exported", dst)
