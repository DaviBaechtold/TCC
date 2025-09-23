#!/usr/bin/env python3
"""
Script principal para treinamento do modelo multimodal.
"""

import argparse
import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from src.models.fusion import MultiModalFusionNetwork
from src.data.loaders import VideoDataLoader
from src.training.trainer import MultiModalTrainer
from src.utils.visualization import visualize_results


from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração do arquivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict[str, Any]) -> MultiModalFusionNetwork:
    """Configura o modelo baseado na configuração."""
    model_config = config['model']
    
    model = MultiModalFusionNetwork(
        depth_model=model_config['depth']['name'],
        segmentation_model=model_config['segmentation']['name'],
        fusion_dim=model_config['fusion']['fusion_dim'],
        output_dim=model_config['fusion']['output_dim'],
        use_temporal=model_config['fusion']['use_temporal']
    )
    
    return model


def setup_data_loader(config: Dict[str, Any]) -> VideoDataLoader:
    """Configura o data loader baseado na configuração."""
    data_config = config['data']
    
    # Atualizar configuração do loader
    loader_config = {
        'batch_size': data_config['loader']['batch_size'],
        'sequence_length': data_config['loader']['sequence_length'],
        'frame_size': tuple(data_config['loader']['frame_size']),
        'num_workers': data_config['loader']['num_workers'],
        'shuffle': data_config['loader']['shuffle']
    }
    
    data_loader = VideoDataLoader()
    data_loader.config.update(loader_config)
    
    return data_loader


def main():
    parser = argparse.ArgumentParser(description='Treinamento do modelo multimodal')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--data_dir', type=str, 
                       help='Diretório com dados de treinamento (não necessário em modo debug)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Diretório para salvar resultados')
    parser.add_argument('--resume', type=str, default=None,
                       help='Caminho para checkpoint para continuar treinamento')
    parser.add_argument('--debug', action='store_true',
                       help='Executar em modo debug com dados sintéticos')
    
    args = parser.parse_args()
    
    # Carregar configuração
    config = load_config(args.config)
    
    # Criar diretórios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar device
    device = torch.device('cpu')  # Forçar CPU para estabilidade inicial
    print(f"Usando device: {device}")
    config['training']['device'] = 'cpu'  # Atualizar config
    
    # Configurar modelo
    print("Configurando modelo...")
    model = setup_model(config)
    model = model.to(device)
    
    # Configurar data loader
    print("Configurando data loader...")
    data_loader_manager = setup_data_loader(config)
    
    if args.debug:
        # Dados sintéticos para debug
        print("Executando em modo debug com dados sintéticos")
        video_paths = ["dummy_video.mp4"]  # Placeholder
        train_loader = None  # Será criado pelo trainer em modo debug
        val_loader = None
    else:
        # Verificar se data_dir foi fornecido
        if not args.data_dir:
            raise ValueError("--data_dir é obrigatório quando não está em modo debug")
            
        # Carregar dados reais
        data_dir = Path(args.data_dir)
        video_paths = list(data_dir.glob("**/*.mp4"))
        
        if not video_paths:
            raise ValueError(f"Nenhum vídeo encontrado em {data_dir}")
        
        print(f"Encontrados {len(video_paths)} vídeos")
        
        # Dividir em treino e validação (80/20)
        split_idx = int(0.8 * len(video_paths))
        train_paths = video_paths[:split_idx]
        val_paths = video_paths[split_idx:]
        
        train_loader = data_loader_manager.create_video_loader(
            [str(p) for p in train_paths]
        )
        val_loader = data_loader_manager.create_video_loader(
            [str(p) for p in val_paths]
        )
    
    # Configurar trainer
    print("Configurando trainer...")
    trainer = MultiModalTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    # Treinar modelo
    print("Iniciando treinamento...")
    if args.debug:
        # Teste rápido com dados sintéticos
        results = trainer.debug_run()
    else:
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_checkpoint=args.resume
        )
    
    # Visualizar resultados
    print("Gerando visualizações...")
    visualize_results(results, 
                     save_path=str(output_dir / "training_results.png"))
    
    print(f"Treinamento concluído! Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()