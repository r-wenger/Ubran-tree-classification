# utils.py
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import params
from datetime import datetime, timedelta
from typing import Optional
import torch.nn.functional as F


def filter_species(data, species_column_name, species_list):
    return data[data[species_column_name].isin(species_list)]


def extract_features_2D_tensor(data, prefix=None):
    if prefix=='S2':
        list_bands = params.list_bands_S2
        dates = params.list_dates_S2
    elif prefix=='Planet':
        prefix = 'PS'
        list_bands = params.list_bands_Planet
        dates = params.list_dates_Planet

    # Fill the tensor with the features values
    if prefix == 'S2' or prefix == 'PS':
        num_samples = len(data.index)
        num_bands = len(list_bands)
        num_dates = len(dates)
        features_tensor = torch.zeros((num_samples, num_bands, num_dates), dtype=torch.float32)
        for j, b in enumerate(list_bands):
            suffix = 'median'
            for k, d in enumerate(dates):
                field_name = f"{prefix}_{b}_{d[4:]}{suffix}"
                if field_name in data.columns:
                    features_tensor[:, j, k] = torch.from_numpy(data[field_name].values).float()
        return features_tensor
    else:
        features_tensor_s2 = extract_features_2D_tensor(data, prefix='S2')
        features_tensor_planet = extract_features_2D_tensor(data, prefix='Planet')
        return features_tensor_s2, features_tensor_planet


# Function to apply t-SNE and visualize the results
def apply_tsne_and_plot(data_tensor, labels, title, output_path):
    # tsne = TSNE(n_components=3, perplexity=40)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=500)
    tsne_results = tsne.fit_transform(data_tensor)
    
    # Creation of the scatter plot for the t-SNE results
    plt.figure(figsize=(10, 8))
    num_classes = len(np.unique(labels))
    
    # Get a color map
    palette = plt.get_cmap('tab20', num_classes)
    
    # Create a patch (label) for each class
    legend_handles = []
    for i in range(num_classes):
        idx = labels == i
        scatter = plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], color=palette(i), label=i, alpha=0.6)
        legend_handles.append(mpatches.Patch(color=palette(i), label=f'Label {i}'))
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=legend_handles, loc="best", title="Labels")
    plt.savefig(output_path)
    plt.close()


def interpolate_time_series(tensor, sensor):    
    batch_size, num_features, sequence_length = tensor.shape # (batch_size, num_bands, num_dates)
        
    # Convert date strings into datetime objects
    if sensor == 'S2':
        dates_datetime = [datetime.strptime(date, '%Y%m%d') for date in params.list_dates_S2]
    elif sensor == 'Planet':
        dates_datetime = [datetime.strptime(date, '%Y%m%d') for date in params.list_dates_Planet]
    
    # Compute the number of days elapsed since the first date
    min_date = dates_datetime[0]
    max_date = dates_datetime[-1]
    total_days = (max_date - min_date).days + 1
    
    # Create the new tensor with the interpolated values
    new_sequence_length = total_days
    new_tensor = torch.zeros((batch_size, num_features, new_sequence_length))
    
    # Convert the original dates to numbers from the first day and fill the new tensor with the original data
    date_nums = np.array([(date - dates_datetime[0]).days for date in dates_datetime])
    new_tensor[:,:,date_nums] = tensor

    # Interpolate the data between the original dates
    for batch in range(batch_size):
        for feature in range(num_features):
            for time in range(sequence_length - 1): # 0 to 21 or 0 to 52
                start = new_tensor[batch, feature, date_nums[time]]
                end = new_tensor[batch, feature, date_nums[time+1]]
                # creation of the new points: num_new_points+2 including start and end, and [1:-1] to remove start and end points to avoid repetition of existing points
                # interpolated = np.linspace(start, end, np.abs(end-start+1))[1:-1]
                interpolated = torch.linspace(start, end, date_nums[time+1]-date_nums[time]+1)[1:-1]
                # interpolated = torch.arange(start, end+1, np.sign(end+1-start))[1:-1]
                new_tensor[batch, feature, date_nums[time]+1: date_nums[time+1]] = interpolated
    return new_tensor, dates_datetime


def interpolate_and_smooth_tensor(data, sensor, window_length=15, polyorder=1):
    """
    Savitsky-Golay
    """
    interpolated_data, dates_datetime = interpolate_time_series(data, sensor)

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    
    # # Start date (February 13, 2022 for S2)
    # start_date = dates_datetime[0]

    # # List of elapsed days
    # days_list = list(range(len(interpolated_data[0,0,:]))) # 0 to end-1

    # # Convert the days to dates
    # dates_list = [start_date + timedelta(days=day) for day in days_list]
        
    ## Plot the original data and interpolated data for a sample
    # plt.figure(figsize=(12, 6))
    # if sensor == 'S2':
    #     plt.plot(dates_datetime, data[0,6,:], label='Original Data', color='b', marker='o')
    #     plt.plot(dates_list, interpolated_data[0,6,:], label='Interpolated Data', color='r', marker='x')
    # elif sensor == 'Planet':
    #     plt.plot(dates_datetime, data[0,3,:], label='Original Data', color='b', marker='o')
    #     plt.plot(dates_list, interpolated_data[0,3,:], label='Interpolated Data', color='r', marker='x')
    # plt.xlabel('Time')
    # plt.ylabel('Reflectance')
    # plt.title(title)
    # plt.legend()
    # plt.savefig(f'{output_path}.png')
    # plt.close()
    
    ## Add Savitsky-Golay smoothing
    smoothed_data = savgol_filter(interpolated_data, window_length=window_length, polyorder=polyorder, axis=2)
    smoothed_data = torch.tensor(smoothed_data)
    
    ## Plot the original data and smoothed data for a sample
    # plt.figure(figsize=(12, 6))
    # if sensor == 'S2':
    #     plt.plot(dates_datetime, data[0,7,:], label='Original Data', color='b', marker='o')
    #     plt.plot(dates_list, smoothed_data[0,7,:], label='Interpolated Data', color='r', marker='x')
    # elif sensor == 'Planet':
    #     plt.plot(dates_datetime, data[0,3,:], label='Original Data', color='b', marker='o')
    #     plt.plot(dates_list, smoothed_data[0,3,:], label='Interpolated Data', color='r', marker='x')
    # plt.xlabel('Time')
    # plt.ylabel('Reflectance')
    # plt.title(title)
    # plt.legend()
    # # output_path = os.path.join(params.path_output, 'train_interpolated_S2_sav.png')
    # plt.savefig(f'{output_path}_sav.png')
    # plt.close()
    return torch.FloatTensor(smoothed_data)


def flatten_tensors_for_tsne(tensor_s2, tensor_planet):
    """
    Aplatit les tenseurs S2 et Planet pour t-SNE.

    Parameters:
    - tensor_s2: Un tenseur pour Sentinel-2 de forme (num_samples, bands_s2, time_s2).
    - tensor_planet: Un tenseur pour Planet de forme (num_samples, bands_planet, time_planet).

    Returns:
    - Un tenseur combiné aplati prêt pour t-SNE de forme (num_samples, num_features_combined).
    """
    # Aplatir les tenseurs le long des dimensions bands et time
    tensor_s2_flat = tensor_s2.reshape(tensor_s2.shape[0], -1)
    tensor_planet_flat = tensor_planet.reshape(tensor_planet.shape[0], -1)
    
    # Concaténer les tenseurs aplatis le long de la dimension des caractéristiques
    tensor_combined_flat = torch.cat((tensor_s2_flat, tensor_planet_flat), dim=1)
    
    return tensor_combined_flat


def global_min_max_normalize(train_tensor, val_tensor, test_tensor, min_val=None, max_val=None):
    """
     Normalise the data of train, val and test for each band
     - concatenates the 3 tensors
     - puts to 0 the nan values
     _ normalises the data with the min and max values
     - splits again the normalized data into train, val and test tensors
    """
    combined_tensor = torch.cat((train_tensor, val_tensor, test_tensor), dim=0)
    combined_tensor[torch.isnan(combined_tensor)] = 0

    if min_val is None and max_val is None:
        min_val = torch.min(combined_tensor)
        max_val = torch.max(combined_tensor)

    combined_tensor = (combined_tensor - min_val) / (max_val - min_val)

    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    print("[INFO] Min and max values for normalization:", min_val, max_val)
    return combined_tensor[:train_size], combined_tensor[train_size:train_size + val_size], combined_tensor[train_size + val_size:]


def global_mean_std_normalize(train_tensor, val_tensor, test_tensor, mean_val=None, std_val=None):
    combined_tensor = torch.cat((train_tensor, val_tensor, test_tensor), dim=0)
    combined_tensor[torch.isnan(combined_tensor)] = 0

    if mean_val is None and std_val is None:
        mean_val = torch.mean(combined_tensor)
        std_val = torch.std(combined_tensor)

    combined_tensor = (combined_tensor - mean_val) / std_val

    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    print("[INFO] Mean and std values for normalization:", mean_val, std_val)
    return combined_tensor[:train_size], combined_tensor[train_size:train_size + val_size], combined_tensor[train_size + val_size:]


def global_normalize(train_tensor, val_tensor, test_tensor, type, min_val=None, max_val=None, mean_val=None, std_val=None):
    if type == 'min_max':
        return global_min_max_normalize(train_tensor, val_tensor, test_tensor, min_val, max_val)
    elif type == 'mean_std':
        return global_mean_std_normalize(train_tensor, val_tensor, test_tensor, mean_val, std_val)


class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class CustomDualDataset(Dataset):
    def __init__(self, features_s2, features_planet, labels, transform=None):
        """
        Initializes the dataset with Sentinel-2 features, Planet features, and labels.
        Args:
            features_s2: Features from Sentinel-2 data.
            features_planet: Features from Planet data.
            labels: Corresponding labels for the features.
            transform: Optional transform to be applied on a sample.
        """
        self.features_s2 = features_s2
        self.features_planet = features_planet
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Assuming that both features have the same number of samples
        return len(self.features_s2)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.
        Args:
            idx: Index of the item to retrieve.
        Returns:
            A tuple of Sentinel-2 features, Planet features, and their label.
        """
        x_s2 = self.features_s2[idx]
        x_planet = self.features_planet[idx]
        y = self.labels[idx]

        if self.transform:
            # Assuming transform can be applied to both sets of features
            x_s2 = self.transform(x_s2)
            x_planet = self.transform(x_planet)

        return (x_s2, x_planet), y


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.relu(self._linear1(x)))


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model). // (K, 1, d_model)
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos
        / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    )
    PE[:, 1::2] = torch.cos(
        pos
        / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    )
    return PE


def generate_local_map_mask(
    chunk_size: int,
    attention_size: int,
    mask_future=False,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)


# Function to get the attention maps by class
def get_attention_maps_by_class(test_dataset, model, inverse_species_mapping, m):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    attention_maps_by_class = {}
    attention_maps_correct = {}
    attention_maps_incorrect = {}
    
    with torch.no_grad():
        for data in test_loader:
            if params.use_multisensors:
                (inputs_s2, inputs_planet), labels = data
                inputs_s2, inputs_planet, labels = inputs_s2.to(device), inputs_planet.to(device), labels.to(device)
                outputs = model(inputs_s2, inputs_planet)
            else:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) # [1, 20]
                
            predictions = torch.argmax(outputs, dim=1) # [1]            
            attention_maps = model.layers_encoding[3].attention_map # [4, 75, 75]
            class_indices = torch.argmax(labels, dim=1) # torch.Size([1]), position of the maximum value in the tensor
            class_id = class_indices.item() # [1]
            
            is_correct = predictions.item() == class_id # True or False
            attention_maps_by_class = attention_maps_correct if is_correct else attention_maps_incorrect
                
            # attention_maps = model.layers_encoding[3].attention_map
            # #Convert one-hot labels to class indices
            # class_indices = torch.argmax(labels, dim=1)
            
            # Iterating over the samples in the batch
            for i in range(class_indices.size(0)):
                class_id = class_indices[i].item()
                if class_id not in attention_maps_by_class:
                    attention_maps_by_class[class_id] = []
                # attention_maps_2 = attention_maps[]
                attention_maps_by_class[class_id].append(attention_maps[i].cpu().numpy())
    
    # average_attention_maps_by_class = compute_average_attention_maps(attention_maps_by_class)
    # plot_attention_maps(average_attention_maps_by_class, inverse_species_mapping, m)
    
    average_attention_maps_correct = compute_average_attention_maps(attention_maps_correct)
    average_attention_maps_incorrect = compute_average_attention_maps(attention_maps_incorrect)
    plot_attention_maps(average_attention_maps_correct, inverse_species_mapping, m, "correct")
    plot_attention_maps(average_attention_maps_incorrect, inverse_species_mapping, m, "incorrect")
            
    # return attention_maps_by_class


# Compute the average attention maps per class
def compute_average_attention_maps(attention_maps_by_class):
    average_attention_maps_by_class = {class_id: np.mean(maps, axis=0) for class_id, maps in attention_maps_by_class.items()}
    return average_attention_maps_by_class


# Function to plot the average attention maps on subplots
def plot_attention_maps(average_attention_maps_by_class, inverse_species_mapping, m, correct):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f'Model {m} - {correct} - {params.use_data}', fontsize=16)  # Titre principal pour la bande spectrale
    
    # Iterating over the classes and their average attention maps
    for class_id, avg_attention_map in average_attention_maps_by_class.items():
        row = class_id // 5
        col = class_id % 5
        ax = axes[row, col]
        im = ax.imshow(avg_attention_map, cmap='plasma')
        ax.set_title(f'Classe {class_id} - {inverse_species_mapping.get(class_id)}')
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, )
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(params.path_output, f'attention_maps_test_classes_{m}_{correct}.png'))


def get_attention_maps_by_head(model, N, h, m):
    fig, axs = plt.subplots(N, h, figsize=(15, 15))
    # Go through the layers and heads to display each attention map
    for i in range(N):
        for j in range(h):
            attention_maps = model.layers_encoding[i].attention_map[j].cpu().detach().numpy()
            ax = axs[i, j]
            # ax.imshow(attention_maps[i, j], cmap='viridis')
            im = ax.imshow(attention_maps, cmap='plasma')
            ax.set_title(f'Layer {i+1}, Head {j+1}')
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(params.path_output, f'attention_maps_{m}.png'))