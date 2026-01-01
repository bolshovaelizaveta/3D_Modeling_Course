import numpy as np
import yaml
import os
from os.path import join, exists
import open3d.ml.torch as ml3d

class POSSDataset(ml3d.datasets.SemanticKITTI):
    def __init__(self, dataset_path, poss_yaml_path, **kwargs):
        super().__init__(dataset_path=dataset_path, name="poss", **kwargs)
        
        # Загружаем карту лейблов
        with open(poss_yaml_path, "r") as f:
            DATA = yaml.safe_load(f)
        
        self.num_classes = 17
        if 'class_weights' in kwargs:
            self.class_weights = np.array(kwargs['class_weights'], dtype=np.float32)
        else:
            self.class_weights = np.ones(17, dtype=np.float32)
            
        # Создаем таблицу перекодировки
        remap_dict = DATA["learning_map"]
        max_key = max(remap_dict.keys()) if remap_dict else 255
        self.remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in remap_dict.items():
            self.remap_lut_val[k] = v

    def get_split_list(self, split):
        # Определяем список секвенций для конкретного сплита
        if split in ['train', 'training']:
            seq_list = self.cfg.training_split
        elif split == 'validation':
            seq_list = self.cfg.validation_split
        elif split == 'test':
            seq_list = self.cfg.test_split
        else:
            seq_list = self.cfg.all_split

        file_list = []
        for seq_id in seq_list:
            # Формируем путь: 
            pc_path = join(self.cfg.dataset_path, 'dataset', 'sequences', str(seq_id), 'velodyne')
            if exists(pc_path):
                files = np.sort(os.listdir(pc_path))
                file_list.append([join(pc_path, f) for f in files if f.endswith('.bin')])
        
        if not file_list:
            return []
        return np.concatenate(file_list, axis=0)

    def get_label_to_names(self):
        # Возвращаем словарь только для 17 классов
        return {i: str(i) for i in range(17)}