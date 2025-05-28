import torch
import torch.nn.functional as F
from src.constants import *
from src.loss import BaseLoss, DDCLoss, CrossEntropyLoss

class SelfEntropyLoss(BaseLoss):
    name = str_self_entropy_loss
    """
    Enhanced Entropy regularization to prevent trivial solution and also penalize harder for imbalanced datasets.
    """

    def __init__(self, model,loss_weight):
        super().__init__(model)
        if loss_weight is not None and str_self_entropy_loss in loss_weight:
            self.weight = loss_weight[str_self_entropy_loss]
        else:
            self.weight = 0.5  # Increase default weight from 0.1 to 0.5
        self.prob_layer = torch.nn.Softmax(dim=1)
        self.target_utilization = 1.0 / self.n_output  # Ideal uniform distribution

    def __call__(self, model):
        eps = 1e-8
        total_loss = 0
        head_losses = []

        for cluster_outputs in model.cluster_outputs:
            # Apply softmax to get cluster probabilities
            cluster_outputs = self.prob_layer(cluster_outputs)

            # Calculate mean probability for each cluster across batch
            prob_mean = cluster_outputs.mean(dim=0)
            prob_mean[(prob_mean < eps).data] = eps
            

            # Traditional entropy loss
            entropy_loss = (prob_mean * torch.log(prob_mean)).sum()

            # Add utilization penalty - KL divergence from uniform distribution
            uniform_target = torch.ones_like(prob_mean) * self.target_utilization
            utilization_loss = torch.sum(uniform_target * torch.log(uniform_target / prob_mean))
            
            # Combined traditional entropy loss and utilization loss
            loss = entropy_loss + utilization_loss

            loss /= model.n_head
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses



class ContrastiveLoss(BaseLoss):
    """\
    Contrastive Loss for multi-modal learning
    Useful for unsupervised clustering when multiple modalities are present
    """
    name = str_contrastive_loss

    def __init__(self, model, loss_weight):
        super().__init__(model)
        if loss_weight is not None:
            if str_contrastive_loss in loss_weight.keys():
                self.weight = loss_weight[str_contrastive_loss]
            else:
                self.weight = 1
        else:
            self.weight = 1
        self.sampling_ratio = 0.25
        self.tau = 0.1
        self.eye = torch.eye(self.n_output, device=model.device_in_use)

    @staticmethod
    def _cosine_similarity(projections):
        h = F.normalize(projections, p=2, dim=1)
        return h @ h.t()

    def _draw_negative_samples(self, predictions, v, pos_indices):
        predictions = torch.cat(v * [predictions], dim=0)
        weights = (1 - self.eye[predictions])[:, predictions[[pos_indices]]].T
        n_negative_samples = int(self.sampling_ratio * predictions.size(0))
        negative_sample_indices = torch.multinomial(
            weights, n_negative_samples, replacement=True
        )
        return negative_sample_indices

    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = torch.diagonal(logits, offset=diagonal_offset)
            _lower = torch.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = torch.arange(0, diag_length)
            _lower_inds = torch.arange(i * n, v * n)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = torch.cat(diagonals, dim=0)
        pos_inds = torch.cat(inds, dim=0)
        return pos, pos_inds

    def __call__(self, model):
        if model.n_modality == 1:
            return 0, [0] * model.n_head

        n_sample = len(model.labels) if model.labels is not None else model.modalities[0].shape[0]

        total_loss = 0
        head_losses = None

        # Create latent projection for contrastive learning
        latent_projection = torch.cat(model.latents, dim=0)
        
        logits = (
            ContrastiveLoss._cosine_similarity(latent_projection)
            / self.tau
        )
        pos, pos_inds = ContrastiveLoss._get_positive_samples(
            logits, model.n_modality, n_sample
        )

        predictions = model.predictions[model.best_head]
        if len(torch.unique(predictions)) > 1:
            neg_inds = self._draw_negative_samples(
                predictions, model.n_modality, pos_inds
            )
            neg = logits[pos_inds.view(-1, 1), neg_inds]
            inputs = torch.cat((pos.view(-1, 1), neg), dim=1)
            labels = torch.zeros(
                model.n_modality * (model.n_modality - 1) * n_sample,
                device=model.device_in_use,
                dtype=torch.long,
            )
            loss = F.cross_entropy(inputs, labels)

            loss /= model.n_head
            loss *= self.weight
        else:
            loss = 0

        total_loss += loss

        return total_loss, head_losses