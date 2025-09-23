"""
Data loaders para diferentes tipos de dados (vídeos, imagens, keypoints).
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import decord
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


class VideoDataset(Dataset):
    """
    Dataset para carregar sequências de vídeo com keypoints e anotações.
    """
    
    def __init__(self,
                 video_paths: List[Union[str, Path]],
                 keypoints_paths: Optional[List[Union[str, Path]]] = None,
                 sequence_length: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 stride: int = 1,
                 transforms=None):
        
        self.video_paths = [Path(p) for p in video_paths]
        self.keypoints_paths = [Path(p) for p in keypoints_paths] if keypoints_paths else None
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.stride = stride
        self.transforms = transforms
        
        if not HAS_OPENCV:
            raise ImportError("opencv-python is required for VideoDataset")
        
        # Preparar índices de sequências
        self.sequence_indices = self._prepare_sequence_indices()
    
    def _prepare_sequence_indices(self) -> List[Tuple[int, int]]:
        """Prepara índices de sequências para cada vídeo."""
        indices = []
        
        for video_idx, video_path in enumerate(self.video_paths):
            # Obter número de frames
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Criar índices de sequências
            for start_frame in range(0, total_frames - self.sequence_length + 1, self.stride):
                indices.append((video_idx, start_frame))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, start_frame = self.sequence_indices[idx]
        
        # Carregar sequência de frames
        frames = self._load_video_sequence(video_idx, start_frame)
        
        # Carregar keypoints se disponível
        keypoints = None
        if self.keypoints_paths:
            keypoints = self._load_keypoints_sequence(video_idx, start_frame)
        
        # Aplicar transformações
        if self.transforms:
            frames = self.transforms(frames)
        
        sample = {
            'frames': frames,
            'video_idx': video_idx,
            'start_frame': start_frame
        }
        
        if keypoints is not None:
            sample['keypoints'] = keypoints
        
        return sample
    
    def _load_video_sequence(self, video_idx: int, start_frame: int) -> torch.Tensor:
        """Carrega sequência de frames de um vídeo."""
        video_path = self.video_paths[video_idx]
        
        try:
            if HAS_DECORD:
                # Usar decord para carregamento eficiente
                vr = decord.VideoReader(str(video_path))
                total_frames = len(vr)
                
                # Ajustar start_frame se necessário
                if start_frame + self.sequence_length > total_frames:
                    start_frame = max(0, total_frames - self.sequence_length)
                
                indices = list(range(start_frame, min(start_frame + self.sequence_length, total_frames)))
                
                # Preencher com último frame se necessário
                while len(indices) < self.sequence_length:
                    indices.append(indices[-1] if indices else 0)
                
                frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
            else:
                # Usar OpenCV
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Ajustar start_frame se necessário
                if start_frame + self.sequence_length > total_frames:
                    start_frame = max(0, total_frames - self.sequence_length)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                frames = []
                for i in range(self.sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        # Repetir último frame se necessário
                        if frames:
                            frame = frames[-1].copy()
                        else:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                cap.release()
                frames = np.array(frames)  # (T, H, W, 3)
            
            # Redimensionar frames
            resized_frames = []
            for frame in frames:
                resized = cv2.resize(frame, self.frame_size)
                resized_frames.append(resized)
            
            frames_array = np.array(resized_frames)  # (T, H, W, 3)
            
            # Converter para tensor e normalizar
            frames_tensor = torch.from_numpy(frames_array).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
            
            return frames_tensor
            
        except Exception as e:
            print(f"Erro ao carregar vídeo {video_path}: {e}")
            # Retornar tensor vazio em caso de erro
            return torch.zeros(self.sequence_length, 3, *self.frame_size)
    
    def _load_keypoints_sequence(self, video_idx: int, start_frame: int) -> torch.Tensor:
        """Carrega sequência de keypoints."""
        if not self.keypoints_paths:
            return None
        
        keypoints_path = self.keypoints_paths[video_idx]
        
        # Carregar keypoints do arquivo (formato a ser definido)
        # Por enquanto, retorna keypoints dummy
        keypoints = torch.randn(self.sequence_length, 33, 3)  # MediaPipe format
        
        return keypoints


class ImageDataset(Dataset):
    """
    Dataset para imagens individuais com anotações.
    """
    
    def __init__(self,
                 image_paths: List[Union[str, Path]],
                 annotations: Optional[List[Dict]] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 transforms=None):
        
        self.image_paths = [Path(p) for p in image_paths]
        self.annotations = annotations or [{}] * len(image_paths)
        self.image_size = image_size
        self.transforms = transforms
        
        if not HAS_OPENCV:
            raise ImportError("opencv-python is required for ImageDataset")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Carregar imagem
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        # Converter para tensor
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # (3, H, W)
        
        # Aplicar transformações
        if self.transforms:
            image_tensor = self.transforms(image_tensor)
        
        sample = {
            'image': image_tensor,
            'annotation': self.annotations[idx],
            'image_path': str(image_path)
        }
        
        return sample


class VideoDataLoader:
    """
    Classe principal para carregamento de dados de vídeo.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Carrega configuração do data loader."""
        default_config = {
            'batch_size': 4,
            'sequence_length': 16,
            'frame_size': (224, 224),
            'num_workers': 4,
            'shuffle': True
        }
        
        if config_path:
            # Carregar configuração do arquivo YAML
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                default_config.update(file_config.get('data_loader', {}))
            except ImportError:
                print("PyYAML not installed, using default config")
            except FileNotFoundError:
                print(f"Config file {config_path} not found, using default config")
        
        return default_config
    
    def create_video_loader(self, 
                           video_paths: List[str],
                           keypoints_paths: Optional[List[str]] = None,
                           transforms=None) -> DataLoader:
        """
        Cria DataLoader para vídeos.
        
        Args:
            video_paths: Lista de caminhos para vídeos
            keypoints_paths: Lista de caminhos para keypoints (opcional)
            transforms: Transformações a aplicar
            
        Returns:
            data_loader: DataLoader configurado
        """
        dataset = VideoDataset(
            video_paths=video_paths,
            keypoints_paths=keypoints_paths,
            sequence_length=self.config['sequence_length'],
            frame_size=self.config['frame_size'],
            transforms=transforms
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'],
            num_workers=self.config['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        return data_loader
    
    def create_image_loader(self,
                           image_paths: List[str],
                           annotations: Optional[List[Dict]] = None,
                           transforms=None) -> DataLoader:
        """
        Cria DataLoader para imagens.
        
        Args:
            image_paths: Lista de caminhos para imagens
            annotations: Lista de anotações (opcional)
            transforms: Transformações a aplicar
            
        Returns:
            data_loader: DataLoader configurado
        """
        dataset = ImageDataset(
            image_paths=image_paths,
            annotations=annotations,
            image_size=self.config['frame_size'],
            transforms=transforms
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'],
            num_workers=self.config['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        return data_loader