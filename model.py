# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import geoopt
from geoopt.manifolds import PoincareBall

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (you can adjust these as needed)
EMBEDDING_DIM = 128
NUM_CLASSES = 10  # Replace with the actual number of classes in your task
CURVATURE_INIT = 1.0  # Initial curvature value


class HyperbolicGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, manifold, **kwargs):
        super(HyperbolicGraphConv, self).__init__(aggr='add', **kwargs)  # "Add" aggregation
        self.manifold = manifold
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # Project input embeddings to the tangent space at the origin
        x_tangent = self.manifold.logmap0(x)
        
        # Apply linear transformation in the tangent space
        x_transformed = self.linear(x_tangent)
        
        # Map back to the manifold
        x_new = self.manifold.expmap0(x_transformed)
        
        # Perform message passing
        out = self.propagate(edge_index, x=x_new)
        
        return out
    
    def message(self, x_j):
        # Messages are the neighbor embeddings
        return x_j
    
    def update(self, aggr_out):
        # Optionally, apply a non-linearity or other operations
        return aggr_out


class HyperbolicGNN(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, manifold):
        super(HyperbolicGNN, self).__init__()
        self.manifold = manifold
        
        # Initialize entity and relation embeddings on the manifold
        self.entity_embeddings = geoopt.ManifoldParameter(
            self.manifold.random(num_entities, embedding_dim), manifold=self.manifold
        )
        self.relation_embeddings = geoopt.ManifoldParameter(
            self.manifold.random(num_relations, embedding_dim), manifold=self.manifold
        )
        
        # Define hyperbolic graph convolution layers
        self.conv1 = HyperbolicGraphConv(embedding_dim, embedding_dim, manifold)
        self.conv2 = HyperbolicGraphConv(embedding_dim, embedding_dim, manifold)
        
        # Define a classifier (for node classification)
        self.classifier = nn.Linear(embedding_dim, NUM_CLASSES)
    
    def forward(self, edge_index):
        x = self.entity_embeddings
        
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Map to tangent space for classification
        x_tangent = self.manifold.logmap0(x)
        
        # Classifier
        out = self.classifier(x_tangent)
        
        return out
    
    def get_entity_embeddings(self):
        return self.entity_embeddings

def hyperbolic_distance(x, y, c=1.0):
    """
    Compute the hyperbolic distance between two points in the Poincaré Ball.

    Args:
        x (torch.Tensor): Tensor of shape (..., dim).
        y (torch.Tensor): Tensor of shape (..., dim).
        c (float): Curvature.

    Returns:
        torch.Tensor: Hyperbolic distances.
    """
    # Ensure inputs are within the Poincaré Ball
    x_norm = torch.clamp(x.norm(p=2, dim=-1, keepdim=True), max=(1 - 1e-5))
    y_norm = torch.clamp(y.norm(p=2, dim=-1, keepdim=True), max=(1 - 1e-5))
    
    diff = x - y
    diff_norm = diff.norm(p=2, dim=-1, keepdim=True)
    
    numerator = 2 * (diff_norm ** 2)
    denominator = (1 - x_norm ** 2) * (1 - y_norm ** 2)
    
    distance = torch.acosh(1 + numerator / denominator)
    return distance.squeeze(-1)


