#!/usr/bin/env python3
"""
Script para avaliação do modelo treinado.
"""

import argparse
import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import numpy as np

from src.models.fusion import MultiModalFusionNetwork
from src.data.loaders import VideoDataLoader
from src.utils.visualization import visualize_results


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> MultiModalFusionNetwork:
    """Carrega modelo do checkpoint."""
    model = MultiModalFusionNetwork(
        depth_model=config['model']['depth']['name'],
        segmentation_model=config['model']['segmentation']['name'],
        fusion_dim=config['model']['fusion']['fusion_dim'],
        output_dim=config['model']['fusion']['output_dim'],
        use_temporal=config['model']['fusion']['use_temporal']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model: MultiModalFusionNetwork, 
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> dict:
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            frames = batch['frames'].to(device)
            keypoints = batch.get('keypoints', None)
            if keypoints is not None:
                keypoints = keypoints.to(device)
            
            # Forward pass
            outputs = model(frames, keypoints, return_intermediate=True)
            
            if isinstance(outputs, tuple):
                latent_features, intermediate = outputs
            else:
                latent_features = outputs
                intermediate = {}
            
            # Adicionar predições
            predictions.append({
                'latent_features': latent_features.cpu().numpy(),
                'intermediate': {k: v.cpu().numpy() for k, v in intermediate.items() if torch.is_tensor(v)}
            })
            
            num_samples += frames.size(0)
    
    results = {
        'num_samples': num_samples,
        'predictions': predictions,
        'avg_feature_magnitude': np.mean([np.linalg.norm(p['latent_features']) for p in predictions])
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Avaliação do modelo multimodal')
    parser.add_argument('--config', type=str, required=True,
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para checkpoint do modelo')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Diretório com dados de teste')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Diretório para salvar resultados')
    parser.add_argument('--debug', action='store_true',
                       help='Executar avaliação com dados sintéticos')
    
    args = parser.parse_args()
    
    # Carregar configuração
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Carregar modelo
    print(f"Carregando modelo de: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Configurar data loader
    if args.debug:
        print("Executando avaliação com dados sintéticos")
        # Dados sintéticos para debug
        results = {
            'debug_mode': True,
            'message': 'Avaliação executada com dados sintéticos'
        }
    else:
        print(f"Carregando dados de: {args.data_dir}")
        data_loader_manager = VideoDataLoader()
        
        data_dir = Path(args.data_dir)
        video_paths = list(data_dir.glob("**/*.mp4"))
        
        if not video_paths:
            raise ValueError(f"Nenhum vídeo encontrado em {data_dir}")
        
        test_loader = data_loader_manager.create_video_loader(
            [str(p) for p in video_paths]
        )
        
        # Avaliar modelo
        print("Iniciando avaliação...")
        results = evaluate_model(model, test_loader, device)
    
    # Salvar resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar resultados detalhados
    results_path = output_dir / 'evaluation_results.npz'
    if not args.debug:
        np.savez(results_path, **results)
        print(f"Resultados salvos em: {results_path}")
    
    # Gerar visualizações
    print("Gerando visualizações...")
    visualize_results(results, 
                     save_path=str(output_dir / "evaluation_results.png"))
    
    print(f"Avaliação concluída! Resultados em: {output_dir}")
    
    if not args.debug:
        print(f"Número de amostras avaliadas: {results['num_samples']}")
        print(f"Magnitude média das features: {results['avg_feature_magnitude']:.4f}")


if __name__ == "__main__":
    main()