import torch
import glob
import os
from mmdet3d.apis import init_model, inference_detector
import mmengine

def run():
    config = 'configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
    ckpts = glob.glob('checkpoints/*nus-3d*.pth')
    if not ckpts:
        print("Ошибка: Чекпоинт nuScenes не найден.")
        return
    checkpoint = ckpts[0]
    pcd = 'demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin'
    
    print(f"Инициализация nuScenes с {checkpoint}...")
    model = init_model(config, checkpoint, device='cpu')
    with torch.no_grad():
        result = inference_detector(model, pcd)
    
    data_sample = result[0] if isinstance(result, (list, tuple)) else result
    out_data = {
        'bboxes_3d': data_sample.pred_instances_3d.bboxes_3d.tensor.cpu().numpy().tolist(),
        'scores_3d': data_sample.pred_instances_3d.scores_3d.cpu().numpy().tolist()
    }
    os.makedirs('outputs/preds', exist_ok=True)
    mmengine.dump(out_data, 'outputs/preds/nuscenes_res.json')
    print(f"nuScenes готов. Найдено: {len(out_data['bboxes_3d'])}")

if __name__ == "__main__":
    run()