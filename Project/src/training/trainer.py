"""
Classe principal de treinamento para o modelo multimodal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class MultiModalTrainer:
    """
    Trainer para o modelo de fusão multimodal.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: torch.device,
                 output_dir: Path):
        
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Configurar otimizador
        self.optimizer = self._setup_optimizer()
        
        # Configurar scheduler
        self.scheduler = self._setup_scheduler()
        
        # Configurar loss functions
        self.criterion = self._setup_loss_functions()
        
        # Estado do treinamento
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Configurar logging
        self._setup_logging()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Configura otimizador baseado na configuração."""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['name'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['name'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Otimizador não suportado: {opt_config['name']}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Configura scheduler de learning rate."""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['name'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['max_epochs'],
                eta_min=1e-7
            )
        elif sched_config['name'].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Configura funções de loss."""
        criteria = {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss(),
            'huber': nn.HuberLoss()
        }
        
        return criteria
    
    def _setup_logging(self):
        """Configura logging com Weights & Biases."""
        logging_config = self.config.get('logging', {})
        use_wandb = logging_config.get('use_wandb', False)
        
        if HAS_WANDB and use_wandb:
            try:
                wandb.init(
                    project=logging_config.get('project_name', 'tcc-multimodal-fusion'),
                    name=logging_config.get('experiment_name', 'baseline'),
                    config=self.config
                )
                self.use_wandb = True
                print("Weights & Biases configurado com sucesso")
            except Exception as e:
                print(f"Erro ao configurar Weights & Biases: {e}")
                print("Continuando sem logging do wandb")
                self.use_wandb = False
        else:
            self.use_wandb = False
            print("Weights & Biases desabilitado")
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computa loss combinado do modelo.
        
        Args:
            outputs: Saídas do modelo
            targets: Targets esperados
            
        Returns:
            total_loss: Loss total
        """
        # Por enquanto, usar MSE simples
        # Em implementação completa, combinar múltiplas losses
        mse_loss = self.criterion['mse'](outputs, targets)
        
        # Adicionar regularização temporal se disponível
        temporal_weight = self.config['training']['loss'].get('temporal_weight', 0.0)
        total_loss = mse_loss
        
        return total_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Treina o modelo por uma época.
        
        Args:
            train_loader: DataLoader de treinamento
            
        Returns:
            avg_loss: Loss média da época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Mover dados para device
            frames = batch['frames'].to(self.device)
            keypoints = batch.get('keypoints', None)
            if keypoints is not None:
                keypoints = keypoints.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(frames, keypoints)
            
            # Criar targets dummy (em implementação real, usar targets reais)
            targets = torch.randn_like(outputs)
            
            # Compute loss
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Atualizar estatísticas
            total_loss += loss.item()
            
            # Atualizar progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log intermediário
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                if self.use_wandb:
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'epoch': self.current_epoch,
                        'step': batch_idx
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Valida o modelo.
        
        Args:
            val_loader: DataLoader de validação
            
        Returns:
            avg_loss: Loss média de validação
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                frames = batch['frames'].to(self.device)
                keypoints = batch.get('keypoints', None)
                if keypoints is not None:
                    keypoints = keypoints.to(self.device)
                
                outputs = self.model(frames, keypoints)
                targets = torch.randn_like(outputs)  # Dummy targets
                
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Salva checkpoint do modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        # Salvar checkpoint regular
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Salvar melhor modelo
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Novo melhor modelo salvo: {best_path}")
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Loop principal de treinamento.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação (opcional)
            resume_checkpoint: Caminho para checkpoint para continuar treinamento
            
        Returns:
            results: Dicionário com resultados do treinamento
        """
        # Carregar checkpoint se fornecido
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        
        max_epochs = self.config['training']['scheduler']['max_epochs']
        save_freq = self.config['training']['save_frequency']
        val_freq = self.config['training']['validation_frequency']
        
        print(f"Iniciando treinamento por {max_epochs} épocas")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Treinamento
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validação
            val_loss = None
            if val_loader and epoch % val_freq == 0:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
            
            # Atualizar scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log da época
            epoch_time = time.time() - start_time
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss_str}, "
                  f"Time = {epoch_time:.2f}s")
            
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'epoch_time': epoch_time,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if val_loss:
                    log_dict['val_loss'] = val_loss
                wandb.log(log_dict)
            
            # Salvar checkpoint
            is_best = val_loss and val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_loss or train_loss, is_best)
        
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'final_epoch': self.current_epoch
        }
        
        return results
    
    def debug_run(self) -> Dict[str, Any]:
        """
        Executa um teste rápido com dados sintéticos.
        
        Returns:
            results: Resultados do teste
        """
        print("Executando teste rápido com dados sintéticos...")
        
        # Criar dados sintéticos
        batch_size = 2
        sequence_length = 8
        frames = torch.randn(batch_size, sequence_length, 3, 224, 224).to(self.device)
        keypoints = torch.randn(batch_size, sequence_length, 99).to(self.device)
        
        # Teste forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(frames, keypoints)
            print(f"Output shape: {outputs.shape}")
        
        # Teste backward pass
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(frames, keypoints)
        targets = torch.randn_like(outputs)
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        print(f"Debug - Loss: {loss.item():.4f}")
        
        results = {
            'debug_loss': loss.item(),
            'output_shape': list(outputs.shape),
            'status': 'success'
        }
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str):
        """Carrega checkpoint para continuar treinamento."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['loss']
        
        print(f"Checkpoint carregado: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")