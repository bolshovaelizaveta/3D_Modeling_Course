import numpy as np
import plotly.graph_objects as go
import mmengine
import os

def get_box_wireframe(box):
    x, y, z, w, l, h, yaw = box[:7]
    
    dx, dy, dz = w/2, l/2, h/2
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
    ])
    
    # Вращение по yaw
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Поворот и сдвиг
    rotated_corners = corners @ R.T + [x, y, z]
    return rotated_corners

def draw_final(name, bin_path, json_path, pcd_dim):
    if not os.path.exists(json_path):
        print(f"Файл {json_path} не найден. Сначала запусти инференс.")
        return

    # Загрузка точек
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, pcd_dim)[:, :3]
    
    # Загрузка результатов
    res = mmengine.load(json_path)
    bboxes = np.array(res['bboxes_3d'])
    scores = np.array(res['scores_3d'])
    
    # Фильтр
    mask = scores > 0.35
    valid_boxes = bboxes[mask]
    obj_count = len(valid_boxes)

    fig = go.Figure()

    step = 5
    fig.add_trace(go.Scatter3d(
        x=points[::step, 0], y=points[::step, 1], z=points[::step, 2],
        mode='markers',
        marker=dict(size=1.2, color=points[::step, 2], colorscale='Viridis', opacity=0.3),
        name='LiDAR Points'
    ))

    # Боксы
    lines_idx = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]]
    
    for box in valid_boxes:
        v = get_box_wireframe(box)
        for start, end in lines_idx:
            fig.add_trace(go.Scatter3d(
                x=[v[start,0], v[end,0]], y=[v[start,1], v[end,1]], z=[v[start,2], v[end,2]],
                mode='lines',
                line=dict(color='#FF3333', width=4),
                showlegend=False
            ))

    fig.update_layout(
        title={
            'text': f"MMDet3D | Dataset: {name} | Objects Detected: {obj_count}",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20)
        },
        template="plotly_dark",
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()

# 1. KITTI (Улица, Детекция машин). python safe_kitti_scannet.py
# draw_final("KITTI", "demo/data/kitti/000008.bin", "outputs/preds/kitti_res.json", pcd_dim=4)

# 2. nuScenes (Улица, Лидар, Детекция). python safe_det_nuscenes.py
# draw_final("nuScenes", "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin", "outputs/preds/nuscenes_res.json", pcd_dim=5)

# 3. ScanNet (Интерьер, Сегментация мебели/стен). python safe_seg_scannet.py
# draw_final("ScanNet", "demo/data/scannet/scene0000_00.bin", "outputs/preds/scannet_res.json", is_seg=True, pcd_dim=6)

# 4. SUN RGB-D (Интерьер, Детекция мебели). python safe_det_sunrgbd.py
# draw_final("SUNRGBD", "demo/data/sunrgbd/sunrgbd_000017.bin", "outputs/preds/sunrgbd_res.json", pcd_dim=4)