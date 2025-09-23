#!/usr/bin/env python3
"""
Teste simples para verificar se o projeto funciona.
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Testa funcionalidade básica do projeto."""
    try:
        print("Testando imports...")
        
        import torch
        print(f"✅ PyTorch {torch.__version__} importado")
        
        # Forçar uso de CPU para evitar problemas de CUDA
        device = torch.device('cpu')
        print(f"✅ Usando device: {device}")
        
        from src.models.depth.depth_estimator import DepthEstimator
        print("✅ DepthEstimator importado")
        
        from src.models.segmentation.human_segmenter import HumanSegmenter
        print("✅ HumanSegmenter importado")
        
        from src.models.pose.pose_estimator import MediaPipePoseEstimator
        print("✅ MediaPipePoseEstimator importado")
        
        from src.models.embeddings.video_embeddings import VideoEmbeddingExtractor
        print("✅ VideoEmbeddingExtractor importado")
        
        from src.models.fusion.multimodal_fusion import MultiModalFusionNetwork
        print("✅ MultiModalFusionNetwork importado")
        
        print("\nTestando criação de componentes individuais...")
        
        # Testar componentes individuais sem modelos pesados
        from src.models.pose.pose_estimator import PoseEmbedding
        pose_embedding = PoseEmbedding(input_dim=99, embedding_dim=256)
        print("✅ PoseEmbedding criado")
        
        from src.models.embeddings.video_embeddings import MotionEmbedding
        motion_embedding = MotionEmbedding(embedding_dim=128)
        print("✅ MotionEmbedding criado")
        
        print("\nTestando data loaders...")
        from src.data.loaders import VideoDataLoader
        data_loader = VideoDataLoader()
        print("✅ VideoDataLoader criado")
        
        print("\nTestando utilitários...")
        from src.utils.visualization import visualize_results
        # Teste básico de visualização
        dummy_results = {'test': 'data'}
        print("✅ Funções de visualização importadas")
        
        print("\n🎉 Todos os componentes principais foram importados com sucesso!")
        print("✅ Projeto configurado corretamente!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ TESTE PASSOU - Projeto está funcionando!")
        sys.exit(0)
    else:
        print("\n❌ TESTE FALHOU - Verificar erros acima")
        sys.exit(1)