from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.constants import *
from src.loss import SelfEntropyLoss, DDCLoss, ContrastiveLoss, CrossEntropyLoss, ReconstructionLoss
from src.scripts import sum_value_lists
import torch.utils.data as D
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from src.scripts import ordered_cmat

class EncoderDecoderOnlySchedule:
    """
    Simplified scheduler for decoder-encoder-only model
    """
    def __init__(self, name, model, method, loss_weight):
        self.name = name
        self.best_loss = np.inf
        self.best_loss_term = None
        
        if name == str_classification:
            self.parameters = chain.from_iterable(
                [
                    model.encoders.parameters(),
                    model.fusers.parameters(),
                    model.projectors.parameters(),
                    model.latent_projector.parameters(),
                    model.clusters.parameters(),
                ]
            )
            self.optimizer = optim.Adam(self.parameters, lr=model.config[str_lr], )
            self.losses = [CrossEntropyLoss(model, loss_weight)]
            self.best_loss_term = str_cross_entropy_loss
        elif name == str_clustering:
            self.parameters = chain.from_iterable(
                [
                    model.encoders.parameters(), # update the encoders parameters during clustering
                    model.latent_projector.parameters(), 
                    model.fusers.parameters(),
                    model.projectors.parameters(),
                    model.clusters.parameters(),
                ]
            )
            self.optimizer = optim.Adam(self.parameters, lr=model.config[str_lr])
            
            # Only use clustering-relevant losses
            self.losses = [
                SelfEntropyLoss(model, loss_weight),
                DDCLoss(model, loss_weight),
                ReconstructionLoss(model, loss_weight),
            ]
            
            # Add contrastive loss if multiple modalities 
            # TODO: do I need this?
            if model.n_modality > 1:
                self.losses.append(ContrastiveLoss(model, loss_weight))
                
            self.best_loss_term = str_ddc_loss
            
        else:
            raise ValueError(f"Unsupported schedule name for encoder-decoder-only model: {name}")

    def step(self, model, train_model):
        if train_model:
            self.optimizer.zero_grad()

        losses = {}
        accumulated_head_losses = []
        total_loss = 0
        for loss in self.losses:
            if loss.name in [str_self_entropy_loss, str_ddc_loss, str_cross_entropy_loss]:
                _, head_losses = loss(model)
                for hd, h_ls in enumerate(head_losses):
                    losses[f'{loss.name}_head_{hd}'] = h_ls
                    total_loss += losses[f'{loss.name}_head_{hd}']
            else:
                losses[loss.name], head_losses = loss(model)
                total_loss += losses[loss.name]

            if train_model and head_losses is not None:
                accumulated_head_losses = sum_value_lists(
                    accumulated_head_losses, head_losses
                )

        if train_model:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, 25)
            self.optimizer.step()
        return losses

    def check_and_save_best_model(self, model, losses, best_model_path, verbose=False):
        if self.best_loss_term is None:
            curr_loss = sum(losses.values())
        else:
            if self.best_loss_term in [str_self_entropy_loss, str_ddc_loss, str_cross_entropy_loss]:
                curr_loss = losses[f'{self.best_loss_term}_head_{model.best_head}']
            else:
                curr_loss = losses[self.best_loss_term]
                
        if curr_loss < self.best_loss:
            model.save_model(best_model_path)
            self.best_loss = curr_loss
            if verbose:
                print("\n")
                print(f"Best model saved with loss: {curr_loss}")
                print(f"Saved at {model.save_path}/{best_model_path}.pt", "\n")


def encoder_decoder_only_run_train(model, dataloader_train, dataloader_val, verbose=False):
    """
    Training function specifically for decoder-encoder-only model
    """
    
    # Force the task to be clustering for encoder-only model
    task = model.config[str_train_task]
    loss_weight = model.config[str_train_loss_weight]
    
    print(f"Training encoder-decoder-only model for {task} classification")
    # Create clustering schedule
    schedule = EncoderDecoderOnlySchedule(task, model, str_train, loss_weight)
    
    for epoch in tqdm(range(model.config[str_train_epochs])):
        epoch += 1
        model.cur_epoch = epoch
        model.train()
        
        # Training step
        encoder_decoder_only_run_through_dataloader(
            model, dataloader_train, schedule, train_model=True
        )

        # Validation step
        model.eval()
        with torch.no_grad():
            encoder_decoder_only_run_through_dataloader(
                model,
                dataloader_val,
                schedule,
                best_model_path=f"{str_train}_{str_best}",
                verbose=verbose
            )
            
        # Save checkpoint
        if epoch % model.config[str_checkpoint] == 0:
            model.save_model(f"{str_train}_epoch_{epoch}")
            if verbose:
                print(f"Model saved at {model.save_path}/{str_train}_epoch_{epoch}.pt", "\n")

        # Evaluate if verbose - USE ENCODER-ONLY EVALUATION
        if verbose:
            metrics = encoder_decoder_only_run_evaluate(model, dataloader_val)
            print("\nEvaluation Metrics:")
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    if value.size > 6:
                        print(f"{key}: Array shape {value.shape}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")


def encoder_decoder_only_run_through_dataloader(
        model,
        dataloader,
        schedule=None,
        train_model=False,
        infer_model=False,
        best_model_path=None,
        give_losses=False,
        verbose=False
):
    from src.scripts import run_through_dataloader
    return run_through_dataloader(model, dataloader, schedule, train_model, infer_model,best_model_path, give_losses, verbose)
    

def encoder_decoder_only_evaluate_outputs(dataloader, outputs):
    """
    Evaluation function specifically for encoder-only models that only computes clustering metrics
    """
    if not isinstance(dataloader.sampler, D.SequentialSampler):
        raise Exception("Please only evaluate outputs with non-shuffling dataloader.")
    
    dataset = dataloader.dataset
    labels = dataset.labels
    labels = labels if type(labels) == np.ndarray else labels.numpy()
    
    # For decoder-encoder-only model, we only care about predictions, not translations
    translations_outputs, predictions, *_ = outputs
    predictions = predictions if type(predictions) == np.ndarray else predictions.numpy()

    # Since we don't have meaningful translations in encoder-only model,
    # we skip R² computation and only compute clustering metrics
    accuracy, conf_mat = ordered_cmat(labels, predictions)
    metrics = {
        "confusion": conf_mat,
        "acc": accuracy,
        "ari": adjusted_rand_score(labels, predictions),
        "nmi": normalized_mutual_info_score(
            labels, predictions, average_method="geometric"
        ),
    }
    return metrics


def encoder_decoder_only_run_evaluate(model, dataloader, give_losses=False, stage='train'):
    """
    Custom evaluation function for encoder-decoder-only models
    """
    model.eval()
    if give_losses:
        losses = {}
        task = model.config[globals()["str_%s_task" % (stage)]]
        loss_weight = model.config[globals()["str_%s_loss_weight" % (stage)]]
        
        schedules = [
            EncoderDecoderOnlySchedule(str_clustering, model, globals()["str_%s" % (stage)], loss_weight)
        ]
        
        with torch.no_grad():
            for ii, schedule in enumerate(schedules):
                loss = encoder_decoder_only_run_through_dataloader(
                    model, dataloader, schedule, infer_model=True, give_losses=give_losses
                )
                losses[f'{task}_schedule{ii}'] = loss
        return losses
    else:
        with torch.no_grad():
            outputs = encoder_decoder_only_run_through_dataloader(model, dataloader, infer_model=True, give_losses=give_losses)
        return encoder_decoder_only_evaluate_outputs(dataloader, outputs)