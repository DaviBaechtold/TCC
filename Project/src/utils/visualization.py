"""
Utilitários de visualização para análise multimodal.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def visualize_results(results: Dict, 
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> None:
    """
    Visualiza resultados do processamento multimodal.
    
    Args:
        results: Dicionário com resultados do modelo
        save_path: Caminho para salvar visualização (opcional)
        show_plot: Se deve mostrar o plot
    """
    if not results:
        print("Nenhum resultado para visualizar")
        return
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análise Multimodal - Resultados', fontsize=16)
    
    # Placeholder para visualizações específicas
    # Em implementação completa, visualizaria cada modalidade
    
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].text(0.5, 0.5, 'Imagem\nOriginal', ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title('Mapa de Profundidade')
    axes[0, 1].text(0.5, 0.5, 'Depth\nMap', ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].axis('off')
    
    axes[0, 2].set_title('Segmentação Humana')
    axes[0, 2].text(0.5, 0.5, 'Human\nSegmentation', ha='center', va='center', transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')
    
    axes[1, 0].set_title('Keypoints')
    axes[1, 0].text(0.5, 0.5, 'MediaPipe\nKeypoints', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    axes[1, 1].set_title('Features Latentes')
    axes[1, 1].text(0.5, 0.5, 'Latent\nFeatures', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    axes[1, 2].set_title('Análise Temporal')
    axes[1, 2].text(0.5, 0.5, 'Temporal\nAnalysis', ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualização salva em: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_depth_map(depth_map: np.ndarray,
                   title: str = "Mapa de Profundidade",
                   colormap: str = 'viridis',
                   save_path: Optional[str] = None) -> None:
    """
    Visualiza mapa de profundidade.
    
    Args:
        depth_map: Array com mapa de profundidade
        title: Título do plot
        colormap: Mapa de cores
        save_path: Caminho para salvar (opcional)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='Profundidade')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_human_segmentation(image: np.ndarray,
                           mask: np.ndarray,
                           alpha: float = 0.6,
                           title: str = "Segmentação Humana") -> None:
    """
    Visualiza segmentação humana sobreposta à imagem.
    
    Args:
        image: Imagem original
        mask: Máscara de segmentação
        alpha: Transparência da máscara
        title: Título do plot
    """
    plt.figure(figsize=(12, 6))
    
    # Imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    # Máscara
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara de Segmentação')
    plt.axis('off')
    
    # Sobreposição
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha, cmap='Reds')
    plt.title('Sobreposição')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_keypoints(image: np.ndarray,
                  keypoints: Dict,
                  connections: Optional[List[Tuple[int, int]]] = None,
                  title: str = "Keypoints MediaPipe") -> None:
    """
    Visualiza keypoints sobre a imagem.
    
    Args:
        image: Imagem de fundo
        keypoints: Dicionário com keypoints
        connections: Lista de conexões entre keypoints
        title: Título do plot
    """
    if not HAS_OPENCV:
        print("OpenCV necessário para visualização de keypoints")
        return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    if 'absolute' in keypoints:
        points = np.array(keypoints['absolute'])
        
        # Plotar pontos
        plt.scatter(points[:, 0], points[:, 1], c='red', s=30, alpha=0.8)
        
        # Plotar conexões se fornecidas
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    x_coords = [points[start_idx, 0], points[end_idx, 0]]
                    y_coords = [points[start_idx, 1], points[end_idx, 1]]
                    plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2)
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_temporal_analysis(features_sequence: np.ndarray,
                          labels: Optional[List[str]] = None,
                          title: str = "Análise Temporal") -> None:
    """
    Visualiza evolução temporal das features.
    
    Args:
        features_sequence: Sequência de features (T, feature_dim)
        labels: Labels para as features
        title: Título do plot
    """
    plt.figure(figsize=(12, 8))
    
    # Reduzir dimensionalidade para visualização se necessário
    if features_sequence.shape[1] > 10:
        # Plotar apenas as primeiras 10 dimensões
        features_to_plot = features_sequence[:, :10]
        if labels is None:
            labels = [f'Feature {i}' for i in range(10)]
    else:
        features_to_plot = features_sequence
        if labels is None:
            labels = [f'Feature {i}' for i in range(features_sequence.shape[1])]
    
    # Plot das features ao longo do tempo
    for i in range(features_to_plot.shape[1]):
        plt.plot(features_to_plot[:, i], label=labels[i], alpha=0.7)
    
    plt.xlabel('Tempo (frames)')
    plt.ylabel('Valor da Feature')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_comparison_plot(results_list: List[Dict],
                          labels: List[str],
                          metric: str = 'accuracy',
                          title: str = "Comparação de Resultados") -> None:
    """
    Cria plot de comparação entre diferentes experimentos.
    
    Args:
        results_list: Lista de dicionários com resultados
        labels: Labels para cada experimento
        metric: Métrica a comparar
        title: Título do plot
    """
    if not results_list:
        print("Nenhum resultado para comparar")
        return
    
    # Extrair valores da métrica
    values = []
    for results in results_list:
        if metric in results:
            values.append(results[metric])
        else:
            values.append(0)
    
    # Criar bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, alpha=0.8)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_visualization_report(results: Dict,
                             save_dir: str,
                             experiment_name: str) -> None:
    """
    Salva relatório completo de visualização.
    
    Args:
        results: Resultados do experimento
        save_dir: Diretório para salvar
        experiment_name: Nome do experimento
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Criar visualização principal
    main_plot_path = save_path / f"{experiment_name}_main.png"
    visualize_results(results, str(main_plot_path), show_plot=False)
    
    print(f"Relatório de visualização salvo em: {save_path}")
    print(f"- Visualização principal: {main_plot_path}")
    
    # Adicionar outras visualizações específicas conforme necessário
    # depth_map, segmentation, keypoints, etc.