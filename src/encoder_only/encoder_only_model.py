from sklearn import preprocessing
import torch
import torch.nn as nn
from src.constants import *
from src.modules import MLP, WeightedMeanFuser, WeightedMeanFeatureFuser, kaiming_init_weights
import numpy as np


class EncoderOnlyModel(nn.Module):
    """
    Simplified encoder-only version of UnitedNet for unsupervised group identification.
    This model only includes encoders, fusers, projectors, and clusters - removing
    decoders, discriminators, and translation components.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_encoder = preprocessing.LabelEncoder()
        
        # Core model parameters
        self.n_head = config[str_n_head]
        self.n_modality = len(config[str_encoders])
        self.noise_level = config.get(str_noise, None)
        
        # Best head tracking
        self.register_buffer(str_best_head, torch.tensor(0, dtype=torch.long))
        self.potential_best_head = []
        self.register_buffer("head_flag", torch.tensor(0, dtype=torch.long))
        
        # Core components for encoding and clustering
        self.encoders = nn.ModuleList(
            [MLP(encoder) for encoder in config[str_encoders]]
        )
        
        self.fusers = nn.ModuleList(
            [WeightedMeanFeatureFuser(self.n_modality, config[str_encoders][0]["output"]) 
             if config[str_fuser_type] == "WeightedFeatureMean" 
             else WeightedMeanFuser(self.n_modality) 
             for _ in range(self.n_head)]
        )
        
        self.projectors = nn.ModuleList(
            [MLP(config[str_projectors]) for _ in range(self.n_head)]
        )
        
        self.clusters = nn.ModuleList(
            [MLP(config[str_clusters]) for _ in range(self.n_head)]
        )
        
        self.prob_layer = torch.nn.Softmax(dim=1)
        
        # Initialize weights
        self.apply(kaiming_init_weights)
        self.train()

    def add_noise(self, inputs, levels, device):
        """Add noise to inputs during training"""
        noised_input = []
        for input, level in zip(inputs, levels):
            shape = input.shape
            m_ = 0
            v_ = torch.var(input).detach() * level
            if v_ > 0:
                noise = torch.normal(m_, v_, size=shape).to(device=device)
                noised_input.append(input + noise)
            else:
                noised_input.append(input)
        return noised_input

    def impute_check(self, orig_modality):
        """Check and impute modalities to match expected dimensions"""
        self.input_dims = [encoder["input"] for encoder in self.config[str_encoders]]
        if type(orig_modality) is not list:
            checked_modalities = []
            for sd in self.input_dims:
                if orig_modality.shape[1] == sd:
                    checked_modalities.append(torch.tensor(orig_modality))
                else:
                    checked_modalities.append(torch.zeros([orig_modality.shape[0], sd]))
        else:
            assert len(orig_modality) == self.n_modality, "please give either full list of all modalities or a single modality"
            checked_modalities = orig_modality
        return checked_modalities

    def forward(self, modalities, labels=None):
        """
        Forward pass for encoder-only model
        Returns: (translations_with_proper_structure, predictions, fused_latents)
        """
        modalities = self.impute_check(modalities)
        modalities = [
            modality.to(device=self.device_in_use) for modality in modalities
        ]

        # Add noise if specified
        if self.noise_level is not None:
            self.modalities = self.add_noise(inputs=modalities, levels=self.noise_level, device=self.device_in_use)
        else:
            self.modalities = modalities

        self.labels = (
            labels.to(device=self.device_in_use) if labels is not None else None
        )

        # Encode each modality
        self.latents = [
            encoder(modality)
            for (encoder, modality) in zip(self.encoders, self.modalities)
        ]

        # Normalize cluster weights
        with torch.no_grad():
            for pt_i in range(self.n_head):
                w = getattr(self.clusters[pt_i], "layers")[0].weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                getattr(self.clusters[pt_i], "layers")[0].weight.copy_(w)

        # Fuse modalities
        self.fused_latents = [fuser(self.latents) for fuser in self.fusers]

        # Project fused representations
        self.hiddens = [
            projector(fused_latent)
            for (projector, fused_latent) in zip(self.projectors, self.fused_latents)
        ]

        # Generate cluster outputs
        self.cluster_outputs = [
            cluster(hidden) for (cluster, hidden) in zip(self.clusters, self.hiddens)
        ]

        # Make predictions
        self.predictions = [
            torch.argmax(self.prob_layer(cluster_outputs), axis=1)
            for cluster_outputs in self.cluster_outputs
        ]

        # Create proper translation structure that matches expected format
        # Each modality should have translations to all modalities (including itself)
        translations_outputs = []
        for i in range(self.n_modality):
            modality_translations = []
            for j in range(self.n_modality):
                if i == j:
                    # Self-reconstruction - use the original modality
                    if self.training:
                        modality_translations.append(self.modalities[i])
                    else:
                        modality_translations.append(self.modalities[i].cpu().numpy())
                else:
                    # Cross-modal translation - use zeros as placeholder
                    if self.training:
                        modality_translations.append(torch.zeros_like(self.modalities[j]))
                    else:
                        modality_translations.append(torch.zeros_like(self.modalities[j]).cpu().numpy())
            translations_outputs.append(modality_translations)
        
        return (
            translations_outputs,  # Proper translation structure
            self.predictions[self.best_head] if self.training else self.predictions[self.best_head].cpu().numpy(),
            self.fused_latents[self.best_head] if self.training else self.fused_latents[self.best_head].cpu().numpy(),
        )

    def save_model(self, filename):
        """Save model to file"""
        if hasattr(self, 'save_path') and self.save_path is not None:
            self.modalities = None
            self.labels = None
            path = f"{self.save_path}/{filename}.pt"
            torch.save(self, path)

    def reset_classify(self):
        """Reset classification components"""
        self.fusers = nn.ModuleList(
            [WeightedMeanFeatureFuser(self.n_modality, self.config[str_encoders][0]["output"]) 
             if self.config[str_fuser_type] == "WeightedFeatureMean" 
             else WeightedMeanFuser(self.n_modality) 
             for _ in range(self.n_head)]
        )
        
        self.projectors = nn.ModuleList(
            [MLP(self.config[str_projectors]) for _ in range(self.n_head)]
        )
        
        self.clusters = nn.ModuleList(
            [MLP(self.config[str_clusters]) for _ in range(self.n_head)]
        )
        
        self.prob_layer = torch.nn.Softmax(dim=1)


class EncoderOnlyUnitedNet:
    """
    Simplified interface for encoder-only UnitedNet model
    """
    def __init__(self, save_path=None, device="mps", technique=None):
        if save_path is not None:
            import os
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.device = device
        self._create_model_for_technique(technique)

    def _set_device(self):
        self.model = self.model.to(device=self.device)
        self.model.device_in_use = self.device

    def _create_model_for_technique(self, technique):
        if technique is None:
            raise ValueError("Please provide a technique configuration")
        self._create_model_from_config(technique)

    def _create_model_from_config(self, config):
        self.model = EncoderOnlyModel(config)
        self.model.save_path = self.save_path
        self._set_device()

    def train(self, adatas_train, save_path=None, adatas_val=None, init_classify=False, verbose=False):
        """Train the encoder-only model"""
        from src.data import create_dataloader
        from src.encoder_only.encoder_only_scheduler import encoder_only_run_train
        from sklearn.utils import class_weight
        import numpy as np
        
        if save_path is not None:
            import os
            os.makedirs(save_path, exist_ok=True)
            self.model.save_path = save_path
            
        if str_label in adatas_train[0].obs.keys():
            labels = adatas_train[0].obs[str_label]
            self.model.class_weights = list(
                class_weight.compute_class_weight(
                    "balanced", classes=np.unique(labels), y=labels
                )
            )
            
        dataloader_train = create_dataloader(
            self.model,
            adatas_train,
            shuffle=True,
            batch_size=self.model.config[str_train_batch_size],
            fit_label=True,
        )
        
        if adatas_val is None:
            adatas_val = adatas_train
            
        dataloader_test = create_dataloader(
            self.model,
            adatas_val,
            shuffle=False,
            batch_size=self.model.config[str_train_batch_size],
        )
        
        if init_classify:
            self.model.reset_classify()
            self._set_device()
            
        encoder_only_run_train(
            self.model, dataloader_train, dataloader_test, verbose=verbose
        )

    def evaluate(self, adatas, give_losses=False, stage='train'):
        """Evaluate the model"""
        from src.data import create_dataloader
        from src.encoder_only.encoder_only_scheduler import encoder_only_run_evaluate
        
        dataloader = create_dataloader(self.model, adatas, shuffle=False)
        return encoder_only_run_evaluate(self.model, dataloader, give_losses=give_losses, stage=stage)

    def infer(self, adatas):
        """Generate inference results using encoder-only model"""
        from src.data import create_dataloader
        
        dataloader = create_dataloader(self.model, adatas, shuffle=False)
        
        # Use encoder-only specific inference
        self.model.eval()
        with torch.no_grad():
            from src.encoder_only.encoder_only_scheduler import encoder_only_run_through_dataloader
            outputs = encoder_only_run_through_dataloader(self.model, dataloader, infer_model=True)
        
        # Create AnnData from outputs
        _, predictions, fused_latents = outputs
        
        import anndata as ad
        import scanpy as sc
        adata = ad.AnnData(fused_latents if type(fused_latents) == np.ndarray else fused_latents.cpu().numpy())
        
        if hasattr(self.model, 'class_weights'):
            adata.obs["predicted_label"] = self.model.label_encoder.inverse_transform(
                predictions.tolist() if type(predictions) == np.ndarray else predictions.cpu().tolist()
            )
        else:
            adata.obs["predicted_label"] = predictions.cpu().tolist() if hasattr(predictions, 'cpu') else predictions.tolist()
            
        adata.obs["predicted_label"] = adata.obs["predicted_label"].astype('category')
        sc.tl.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=30)
        sc.tl.umap(adata)
        return adata

    def predict_label(self, adatas):
        """Predict labels for input data using encoder-only model"""
        from src.data import create_dataloader
        
        dataloader = create_dataloader(self.model, adatas, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            from src.encoder_only.encoder_only_scheduler import encoder_only_run_through_dataloader
            outputs = encoder_only_run_through_dataloader(self.model, dataloader, infer_model=True)
        
        # Return just the predictions
        return outputs[1]  # predictions are the second element

    def load_model(self, path, device='mps'):
        """Load a saved model"""
        import torch
        self.model = torch.load(path, map_location=torch.device(device), weights_only=False)