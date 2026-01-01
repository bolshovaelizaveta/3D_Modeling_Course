import os
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

PRIMITIVE_CLASSES = {
    "box": 1,
    "sphere": 2,
    "cylinder": 3,
    "torus": 4,
    "cone": 5,
}

def create_primitive(primitive_type, size_range=(0.2, 0.6), height_range=(0.2, 0.8)):
    s_min, s_max = size_range
    h_min, h_max = height_range

    if primitive_type == "box":
        sx, sy, sz = np.random.uniform(s_min, s_max, 3)
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    elif primitive_type == "sphere":
        r = np.random.uniform(s_min, s_max) * 0.5
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    elif primitive_type == "cylinder":
        r = np.random.uniform(s_min, s_max) * 0.3
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h)
    elif primitive_type == "torus":
        r_torus = np.random.uniform(s_min, s_max) * 0.3
        r_tube = r_torus * 0.3
        mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=r_torus, tube_radius=r_tube)
    elif primitive_type == "cone":
        r = np.random.uniform(s_min, s_max) * 0.4
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=r, height=h)
    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.random.uniform(0.2, 0.9, size=3))
    return mesh

def random_transform(mesh, scene_extent=2.0, max_rotation_deg=45.0):
    tx, ty = np.random.uniform(-scene_extent, scene_extent, 2)
    tz = np.random.uniform(-scene_extent * 0.1, scene_extent * 0.1)
    
    angle = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    R[:2, :2] = np.array([[cos_a, sin_a], [-sin_a, cos_a]]) # Z rotation
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return mesh.transform(T)

def sample_points_from_mesh(mesh, num_points=4096):
    pcd = mesh.sample_points_poisson_disk(num_points)
    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)))
    return pcd

def build_scene(num_objects, points_per_object, scene_extent, primitive_types):
    all_points, all_colors, all_semantic, all_instance, bboxes = [], [], [], [], []

    for inst_id in range(num_objects):
        p_type = np.random.choice(primitive_types)
        mesh = random_transform(create_primitive(p_type), scene_extent)
        pcd = sample_points_from_mesh(mesh, num_points=points_per_object)

        pts = np.asarray(pcd.points)
        all_points.append(pts)
        all_colors.append(np.asarray(pcd.colors))
        
        cls_id = PRIMITIVE_CLASSES[p_type]
        all_semantic.append(np.full(len(pts), cls_id, dtype=np.int32))
        all_instance.append(np.full(len(pts), inst_id, dtype=np.int32))

        aabb = pcd.get_axis_aligned_bounding_box()
        bbox = np.zeros(7, dtype=np.float32)
        bbox[0:3] = aabb.get_center()
        bbox[3:6] = aabb.get_extent()
        bbox[6] = cls_id # Label
        bboxes.append(bbox)

    if not all_points: return None
    
    vert = np.concatenate([np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)], axis=1)
    return (vert, 
            np.concatenate(all_semantic, axis=0), 
            np.concatenate(all_instance, axis=0), 
            np.stack(bboxes, axis=0))

def generate_dataset(output_dir, num_train=50, num_val=10, num_test=10, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {"train": num_train, "val": num_val, "test": num_test}
    
    idx = 0
    for split, count in splits.items():
        with open(output_dir / f"{split}_scenes.txt", "w") as f:
            for _ in range(count):
                scene_id = f"scene{idx:04d}_00"
                print(f"Generating {split} scene {scene_id}...")
                
                n_obj = np.random.randint(kwargs.get('min_objects', 3), kwargs.get('max_objects', 6) + 1)
                data = build_scene(n_obj, kwargs.get('points_per_object', 2048), kwargs.get('scene_extent', 2.0), list(PRIMITIVE_CLASSES.keys()))
                
                if data:
                    vert, sem, ins, bbox = data
                    np.save(output_dir / f"{scene_id}_vert.npy", vert.astype(np.float32))
                    np.save(output_dir / f"{scene_id}_sem_label.npy", sem.astype(np.int32))
                    np.save(output_dir / f"{scene_id}_ins_label.npy", ins.astype(np.int32))
                    np.save(output_dir / f"{scene_id}_bbox.npy", bbox.astype(np.float32))
                    f.write(f"{scene_id}\n")
                idx += 1

if __name__ == "__main__":
    generate_dataset("data/scannet_primitives", num_train=100, num_val=20, num_test=20)