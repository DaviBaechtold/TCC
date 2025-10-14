#!/usr/bin/env python3
"""Lista todas as câmeras disponíveis no sistema."""

import cv2


def list_cameras(max_test=10):
    """
    Testa índices de 0 a max_test e lista câmeras disponíveis.
    
    Args:
        max_test: Número máximo de índices para testar
    """
    available_cameras = []
    
    print("🎥 Procurando câmeras disponíveis...\n")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Tentar obter nome da câmera (nem sempre disponível)
                backend = cap.getBackendName()
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend
                })
                
                print(f"✅ Câmera {i} encontrada:")
                print(f"   Resolução: {width}x{height}")
                print(f"   FPS: {fps}")
                print(f"   Backend: {backend}")
                print()
            
            cap.release()
    
    if not available_cameras:
        print("❌ Nenhuma câmera encontrada!")
        print("\nDicas:")
        print("- Verifique se a câmera está conectada")
        print("- Verifique permissões: ls -l /dev/video*")
        print("- Adicione seu usuário ao grupo video: sudo usermod -aG video $USER")
    else:
        print(f"\n📊 Total: {len(available_cameras)} câmera(s) disponível(is)")
        print("\n💡 Para usar uma câmera específica, use:")
        print("   python src/evaluation/run_realtime.py --source <INDEX> ...")
        print("\nExemplos:")
        for cam in available_cameras:
            print(f"   --source {cam['index']}  # Câmera {cam['index']} ({cam['width']}x{cam['height']})")
    
    return available_cameras


def test_camera_preview(camera_index=0, duration=5):
    """
    Abre preview de uma câmera específica.
    
    Args:
        camera_index: Índice da câmera
        duration: Duração em segundos (0 = até pressionar 'q')
    """
    print(f"\n🎬 Abrindo preview da câmera {camera_index}...")
    print("Pressione 'q' para sair")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Erro ao abrir câmera {camera_index}")
        return False
    
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Erro ao capturar frame")
            break
        
        # Adicionar info no frame
        cv2.putText(frame, f"Camera {camera_index}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"Camera {camera_index} Preview", frame)
        
        # Sair com 'q' ou após duração
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if duration > 0 and (time.time() - start_time) > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Preview encerrado")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lista e testa câmeras disponíveis")
    parser.add_argument("--max", type=int, default=10,
                        help="Número máximo de índices para testar (padrão: 10)")
    parser.add_argument("--preview", type=int, default=None,
                        help="Testar preview de uma câmera específica")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duração do preview em segundos (0 = até pressionar 'q')")
    
    args = parser.parse_args()
    
    if args.preview is not None:
        test_camera_preview(args.preview, args.duration)
    else:
        list_cameras(args.max)
