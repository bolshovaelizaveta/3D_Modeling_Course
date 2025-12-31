import torch
from mmdet3d.apis import init_model, inference_detector
import mmengine
import os

def run():
    config = 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
    checkpoint = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
    pcd = 'demo/data/kitti/000008.bin'
    
    print("Инициализация KITTI...")
    model = init_model(config, checkpoint, device='cpu')
    with torch.no_grad():
        result = inference_detector(model, pcd)
    
    data_sample = result[0] if isinstance(result, (list, tuple)) else result
    
    out_data = {
        'bboxes_3d': data_sample.pred_instances_3d.bboxes_3d.tensor.cpu().numpy().tolist(),
        'scores_3d': data_sample.pred_instances_3d.scores_3d.cpu().numpy().tolist()
    }
    
    os.makedirs('outputs/preds', exist_ok=True)
    mmengine.dump(out_data, 'outputs/preds/kitti_res.json')
    print(f"KITTI готов. Найдено объектов: {len(out_data['bboxes_3d'])}")

if __name__ == "__main__":
    run()