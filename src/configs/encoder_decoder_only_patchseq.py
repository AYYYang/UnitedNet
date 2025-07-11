from src.constants import *

# Encoder-only configuration for PatchSeq
encoder_decoder_only_patchseq_config = {
    "train_batch_size": 16,
    "finetune_batch_size": 16,
    "transfer_batch_size": None,
    "train_epochs": 20,
    "finetune_epochs": 20,
    "transfer_epochs": None,
    "train_task": str_classification,
    "finetune_task": str_classification,
    "transfer_task": None,
    
    # Loss weights optimized for encoder-only clustering
    "train_loss_weight": {
        str_self_entropy_loss: 0.05,    
        str_ddc_loss: 0.1,             
        str_contrastive_loss: 0.1,     
        str_reconstruction_loss: 1.0, 
    },
    "finetune_loss_weight": {
        str_self_entropy_loss: 0.05,   # Reduce entropy regularization in fine-tuning
        str_ddc_loss: 1.0,
        str_contrastive_loss: 0.3,
    },
    "transfer_loss_weight": None,
    
    "lr": 0.001,
    "checkpoint": 10,  # Save checkpoints more frequently
    "n_head": 10,
    "fuser_type": "WeightedFeatureMean",
    "noise_level": [0, 0, 0.01],  # Light noise on morphology only
    
    # Encoders - same as original but optimized for clustering
    "encoders": [
        {
            # RNA encoder
            "input": 1252,
            "hiddens": [1024, 512, 256, 128],
            "output": 68,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0.3, 0.1, 0, 0, 0],  # Reduced dropout for better feature learning
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, True],
            "is_binary_input": False,
        },
        {
            # Ephys encoder
            "input": 68,
            "hiddens": [512, 256, 128, 64],  # Slightly smaller since input is smaller
            "output": 68,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0, 0, 0, 0, 0],
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, True],
            "is_binary_input": False,
        },
        {
            # Morphology encoder
            "input": 514,
            "hiddens": [512, 256, 128, 64],
            "output": 68,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0.1, 0, 0, 0, 0],
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, True],
            "is_binary_input": False,
        },
    ],
    
    # Projectors - maps fused features to clustering space
    # TODO: might add another layer?
    "projectors": {
        "input": 68,
        "hiddens": [128],  # Added hidden layer for better representation
        "output": 100,
        "use_biases": [True, True],
        "dropouts": [0.1, 0],
        "activations": ['relu', None],
        "use_batch_norms": [False, False],
        "use_layer_norms": [True, False],
    },
    
    # Clusters - final clustering layer
    "clusters": {
        "input": 100,
        "hiddens": [],
        "output": 27,  # Number of cell types in PatchSeq
        "use_biases": [False],
        "dropouts": [0],
        "activations": [None],
        "use_batch_norms": [False],
        "use_layer_norms": [False],
    },

    "latent_projector": {
        "input": 68,
        "hiddens": [128],
        "output": 64,  # Smaller projection for contrastive learning
        "use_biases": [True, True],
        "dropouts": [0, 0],
        "activations": ["relu", None],
        "use_batch_norms": [False, False],
        "use_layer_norms": [True, False],
        "is_binary_input": False,
    },
    # Decoders
    "decoders": [
        {
            "input": 68,
            "hiddens": [128, 256, 512, 1024],
            "output": 1252,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0, 0, 0, 0, 0],
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, False],
        },
        {
            "input": 68,
            "hiddens": [128, 256, 512, 1024],
            "output": 68,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0, 0, 0, 0, 0],
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, False],
        },
        {
            "input": 68,
            "hiddens": [128, 256, 512, 1024],
            "output": 514,
            "use_biases": [True, True, True, True, True],
            "dropouts": [0, 0, 0, 0, 0],
            "activations": ["relu", "relu", "relu", "relu", None],
            "use_batch_norms": [False, False, False, False, False],
            "use_layer_norms": [False, False, False, False, False],
        },
    ],
}