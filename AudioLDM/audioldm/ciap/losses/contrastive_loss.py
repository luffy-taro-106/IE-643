import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, image_embeddings, audio_embeddings):
        """
        Compute the contrastive loss between image and audio embeddings.

        Args:
            image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, embedding_dim).
            audio_embeddings (torch.Tensor): Audio embeddings of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, audio_embeddings.T) / self.temperature

        # Create labels (diagonal is positive pair)
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size).to(image_embeddings.device)

        # Compute cross-entropy loss
        loss_image_to_audio = F.cross_entropy(logits, labels)
        loss_audio_to_image = F.cross_entropy(logits.T, labels)

        # Average the two losses
        loss = (loss_image_to_audio + loss_audio_to_image) / 2
        return loss