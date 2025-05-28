from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.constants import *
from src.encoder_only.encoder_only_loss import SelfEntropyLoss, DDCLoss, ContrastiveLoss, CrossEntropyLoss
import torch.utils.data as D
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from src.scripts import ordered_cmat

class EncoderOnlySchedule:
    """
    Simplified scheduler for encoder-only model that only uses clustering losses
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
            ]
            
            # Add contrastive loss if multiple modalities
            if model.n_modality > 1:
                self.losses.append(ContrastiveLoss(model, loss_weight))
                
            self.best_loss_term = str_ddc_loss
            
        else:
            raise ValueError(f"Unsupported schedule name for encoder-only model: {name}")

    def step(self, model, train_model):
        if train_model:
            self.optimizer.zero_grad()

        losses = {}
        total_loss = 0
        
        for loss in self.losses:
            if loss.name in [str_self_entropy_loss, str_ddc_loss]:
                _, head_losses = loss(model)
                for hd, h_ls in enumerate(head_losses):
                    losses[f'{loss.name}_head_{hd}'] = h_ls
                    total_loss += losses[f'{loss.name}_head_{hd}']
            else:
                losses[loss.name], _ = loss(model)
                total_loss += losses[loss.name]

        if train_model:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, 25)
            self.optimizer.step()
            
        return losses

    def check_and_save_best_model(self, model, losses, best_model_path, verbose=False):
        if self.best_loss_term is None:
            curr_loss = sum(losses.values())
        else:
            if self.best_loss_term in [str_self_entropy_loss, str_ddc_loss]:
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


def encoder_only_run_train(model, dataloader_train, dataloader_val, verbose=False):
    """
    Training function specifically for encoder-only model
    """
    print("Training encoder-only model for unsupervised clustering")
    
    # Force the task to be clustering for encoder-only model
    model.config[str_train_task] = str_unsupervised_group_identification
    
    task = model.config[str_train_task]
    loss_weight = model.config[str_train_loss_weight]
    
    # Create clustering schedule
    schedule = EncoderOnlySchedule(str_clustering, model, str_train, loss_weight)
    
    for epoch in tqdm(range(model.config[str_train_epochs])):
        epoch += 1
        model.cur_epoch = epoch
        model.train()
        
        # Training step
        encoder_only_run_through_dataloader(
            model, dataloader_train, schedule, train_model=True
        )

        # Validation step
        model.eval()
        with torch.no_grad():
            encoder_only_run_through_dataloader(
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
            metrics = encoder_only_run_evaluate(model, dataloader_val)
            print("\nEvaluation Metrics:")
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    if value.size > 6:
                        print(f"{key}: Array shape {value.shape}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")


def encoder_only_run_through_dataloader(
        model,
        dataloader,
        schedule=None,
        train_model=False,
        infer_model=False,
        best_model_path=None,
        give_losses=False,
        verbose=False
):
    """
    Simplified dataloader runner for encoder-only model
    """
    from src.scripts import (
        amplify_value_dictionary_by_sample_size,
        sum_value_dictionaries,
        average_dictionary_values_by_sample_size,
        inplace_combine_tensor_lists,
        concat_tensor_lists
    )
    
    all_outputs = []
    all_losses = {}

    for modalities, labels in dataloader:
        outputs = model(modalities, labels)
        
        if schedule is not None:
            losses = schedule.step(model, train_model)
            losses = amplify_value_dictionary_by_sample_size(losses, len(labels))
            all_losses = sum_value_dictionaries(all_losses, losses)

        if infer_model:
            inplace_combine_tensor_lists(all_outputs, outputs)

    if all_losses:
        all_losses = average_dictionary_values_by_sample_size(
            all_losses, len(dataloader.dataset)
        )
        
        # Handle best head selection for clustering losses
        for ls_name in all_losses:
            if ('ddc' in ls_name) or ('self_entropy' in ls_name):
                ls_name_hd = '_'.join(ls_name.split('_')[:-1])
                head_losses = {k: all_losses[k] for k in all_losses.keys() if ls_name_hd in k}
                current_best_head = torch.tensor(int(min(head_losses, key=head_losses.get).split('_')[-1]))
                if hasattr(model, 'potential_best_head'):
                    model.potential_best_head.append(current_best_head)
                model.best_head = current_best_head
                break

    if best_model_path is not None:
        if len(model.potential_best_head) > 0:
            cur_bc_heads = torch.bincount(torch.tensor(model.potential_best_head))
            if any(cur_bc_heads >= model.config[str_train_epochs]//3):
                model.best_head = torch.argmax(cur_bc_heads)
        else:
            model.best_head = torch.tensor(0, dtype=torch.long)
        schedule.check_and_save_best_model(model, all_losses, best_model_path, verbose=verbose)
        
    if give_losses:
        assert len(all_losses.keys()) > 0, 'Losses are empty'
        return all_losses
    else:
        return concat_tensor_lists(all_outputs) if all_outputs else []
    


def encoder_only_evaluate_outputs(dataloader, outputs):
    """
    Evaluation function specifically for encoder-only models that only computes clustering metrics
    """
    if not isinstance(dataloader.sampler, D.SequentialSampler):
        raise Exception("Please only evaluate outputs with non-shuffling dataloader.")
    
    dataset = dataloader.dataset
    labels = dataset.labels
    labels = labels if type(labels) == np.ndarray else labels.numpy()
    
    # For encoder-only model, we only care about predictions, not translations
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


def encoder_only_run_evaluate(model, dataloader, give_losses=False, stage='train'):
    """
    Custom evaluation function for encoder-only models
    """
    model.eval()
    if give_losses:
        losses = {}
        task = model.config[globals()["str_%s_task" % (stage)]]
        loss_weight = model.config[globals()["str_%s_loss_weight" % (stage)]]
        
        schedules = [
            EncoderOnlySchedule(str_clustering, model, globals()["str_%s" % (stage)], loss_weight)
        ]
        
        with torch.no_grad():
            for ii, schedule in enumerate(schedules):
                loss = encoder_only_run_through_dataloader(
                    model, dataloader, schedule, infer_model=True, give_losses=give_losses
                )
                losses[f'{task}_schedule{ii}'] = loss
        return losses
    else:
        with torch.no_grad():
            outputs = encoder_only_run_through_dataloader(model, dataloader, infer_model=True, give_losses=give_losses)
        return encoder_only_evaluate_outputs(dataloader, outputs)