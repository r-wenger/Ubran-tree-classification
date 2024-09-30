# main.py
# Standard library imports
import os
import shutil

# Data manipulation and numerical computing
import numpy as np
import pandas as pd

# Machine learning and data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Deep Learning with PyTorch
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold

# Geospatial data processing and visualization
import geopandas as gpd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Other imports
from datetime import datetime
import time
import re
from codecarbon import OfflineEmissionsTracker

# Other project files
import params
from utils import CustomDataset, CustomDualDataset, extract_features_2D_tensor, global_normalize, filter_species, apply_tsne_and_plot, flatten_tensors_for_tsne, interpolate_time_series, interpolate_and_smooth_tensor, get_attention_maps_by_class, get_attention_maps_by_head
import test
from train import TrainModel
from models import InceptionTime, DualInceptionTime, HybridInceptionTime, DualHybridInceptionTime, Transformer, DualTransformer, TSTransformerEncoderClassiregressor, TSTransformerEncoderClassiregressorDual, LITE, DualLITE


def main():
    #CARBON Tracker
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()

    start = time.time()

    # Creation of the output folders and deletion of the previous ones if it exists
    # paths = [params.path_output, params.gradcam_boxplot_dir, params.model_path]
    paths = [params.path_output, params.model_path]
    for p in paths:
        if os.path.exists(p):
            shutil.rmtree(p)
            os.makedirs(p)
        else:
            os.makedirs(p)
    
    # General information
    print("[INFO] General information parameters",
            "\n Scenar:", params.name_scenar,
            "\n City:", params.city,
            "\n Model:", params.name_model, 
            "\n Number of classes:", params.num_classes, 
            "\n Number of splits:", params.n_splits,
            "\n Batch size:", params.batch_size,
            "\n Number of epochs:", params.num_epochs,
            "\n Learning rate:", params.learning_rate,
            "\n Data used:", params.use_data,
            "\n Use t-SNE:", params.tsne,
            "\n Kernel sizes S2:", params.kernel_sizes_s2 if re.search(r'S2', params.use_data) and not re.search(r'Transformer', params.name_model) else None,
            "\n Kernel sizes Planet:", params.kernel_sizes_planet if re.search(r'Planet', params.use_data) and not re.search(r'Transformer', params.name_model) else None,
            "\n Model dimension:", params.d_model if re.search(r'Transformer', params.name_model) else None,
            "\n Number of heads:", params.h if re.search(r'Transformer', params.name_model) else None,
            "\n Number of layers:", params.N if re.search(r'Transformer', params.name_model) else None,
            "\n Attention size:", params.attention_size if re.search(r'Transformer', params.name_model) else None,
            "\n Dropout:", params.dropout,
            "\n Query size:", params.q if re.search(r'Transformer', params.name_model) and params.transf == 1 else None,
            "\n Value size:", params.v if re.search(r'Transformer', params.name_model) and params.transf == 1 else None,
            "\n Feedforward dimension:", params.dim_feedforward if re.search(r'Transformer', params.name_model) and params.transf == 2 else None,
            "\n Normalization layer:", params.normalization_layer if re.search(r'Transformer', params.name_model) and params.transf == 2 else None,
            "\n Interpolate:", params.interpolate,
            "\n"
            )
            
    # Setting the device to GPU if available
    print("[INFO] Initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device set to {device}")
    
    # kfold = KFold(n_splits=params.n_splits, shuffle=True, random_state=11)
    stratified_kfold = StratifiedKFold(n_splits=params.n_splits, shuffle=True, random_state=11)
    
    # Read the Shapefile using GeoPandas and filter the data according to the wanted species (10 or 20)
    print("[INFO] Shapefile loading...")
    start1 = time.time()
    data = gpd.read_file(params.shapefile_path, engine="pyogrio")
    print("[INFO] Shapefile loaded in", (time.time()-start1), "seconds.")
    print("[INFO] Data shape:", data.shape)
    
    data = filter_species(data, params.libelle, params.selected_species)
    
    # Labels
    labels = data[params.libelle]
    
    # Splitting the data
    print("[INFO] Splitting data...")
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=11, stratify=labels) # train: 85%, test: 15%
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=11, stratify=train_data[params.libelle]) #0.25 train: 60%, val: 25%
    print(f"[INFO] Data split into training ({len(train_data)/len(data)*100:.2f}%), validation ({len(val_data)/len(data)*100:.2f}%), and test ({len(test_data)/len(data)*100:.2f}%) sets.")
    
    # Check the distribution of classes in each set
    print("Distribution of classes in the original dataset:", data[params.libelle].value_counts())
    print("Distribution of classes in the training dataset:", train_data[params.libelle].value_counts())
    print("Distribution of classes in the validation dataset:", val_data[params.libelle].value_counts())
    print("Distribution of classes in the test dataset:", test_data[params.libelle].value_counts())
    
    # List of species present in the Shapefile
    especes = data[params.libelle].unique()

    # Encode the species labels into integers
    species_mapping = {species: i for i, species in enumerate(especes)}    
    print("Species mapping:")
    print(json.dumps(species_mapping, indent=4, ensure_ascii=False))
    inverse_species_mapping = {v: j for j, v in species_mapping.items()}
    train_labels = np.array([species_mapping[species] for species in train_data[params.libelle]])
    val_labels = np.array([species_mapping[species] for species in val_data[params.libelle]])
    test_labels = np.array([species_mapping[species] for species in test_data[params.libelle]])
    print("[INFO] Data filtered by selected species. \n")
    
    # Binaries and convert labels to PyTorch tensors
    label_binarizer = LabelBinarizer() # binarize en one-hot encoder
    train_labels_tensor = torch.tensor(label_binarizer.fit_transform(train_labels)).float() # size (n, 20)
    val_labels_tensor = torch.tensor(label_binarizer.transform(val_labels)).float()
    test_labels_tensor = torch.tensor(label_binarizer.transform(test_labels)).float()
    train_val_labels_tensor = torch.cat((train_labels_tensor, val_labels_tensor), dim=0)
    train_val_labels = np.concatenate((train_labels, val_labels), axis=0)
    # print("Label size:", train_labels_tensor.shape, val_labels_tensor.shape, test_labels_tensor.shape)
        
    # Load Data and extract features from data
    if not params.use_multisensors:
        print(f"[INFO] Loading {params.use_data} data...")
        train_tensor = extract_features_2D_tensor(train_data, prefix=params.use_data)
        print("Dimensions of the training tensor:", train_tensor.shape)
        val_tensor = extract_features_2D_tensor(val_data, prefix=params.use_data)
        print("Dimensions of the validation tensor:", val_tensor.shape)
        test_tensor = extract_features_2D_tensor(test_data, prefix=params.use_data)
        print("Dimensions of the test tensor:", test_tensor.shape)
    else:
        print("[INFO] Loading Sentinel-2 and Planet data...")
        train_tensor_s2, train_tensor_planet = extract_features_2D_tensor(train_data)
        val_tensor_s2, val_tensor_planet = extract_features_2D_tensor(val_data)
        test_tensor_s2, test_tensor_planet = extract_features_2D_tensor(test_data)
        print("S2 - Dimensions of the training tensor:", train_tensor_s2.shape)
        print("S2 - Dimensions of the validation tensor:", val_tensor_s2.shape)
        print("S2 - Dimensions of the test tensor:", test_tensor_s2.shape)
        print("Planet - Dimensions of the training tensor:", train_tensor_planet.shape)
        print("Planet - Dimensions of the validation tensor:", val_tensor_planet.shape)
        print("Planet - Dimensions of the test tensor:", test_tensor_planet.shape)

    # Apply t-SNE if specified and using multisensors
    if params.tsne:# and params.use_multisensors:
        print("[INFO] Applying t-SNE...")
        # Put Nan values to 0: precaution to take
        train_tensor_s2[torch.isnan(train_tensor_s2)] = 0
        train_tensor_planet[torch.isnan(train_tensor_planet)] = 0
        
        test_tensor_planet[torch.isnan(test_tensor_planet)] = 0
        test_tensor_s2[torch.isnan(test_tensor_s2)] = 0
        
        # Flatten the tensors for the training and test set (TSNE requires flatten tensors)
        train_tensor_combined = flatten_tensors_for_tsne(train_tensor_s2, train_tensor_planet)
        test_tensor_combined = flatten_tensors_for_tsne(test_tensor_s2, test_tensor_planet)

        scaler = StandardScaler()
        train_tensor_combined = scaler.fit_transform(train_tensor_combined)
        test_tensor_combined = scaler.fit_transform(test_tensor_combined)

        print("[INFO] Applying t-SNE for S2 and training set...")
        output_path = os.path.join(params.path_output, 't-SNE_train.png')
        apply_tsne_and_plot(train_tensor_combined, train_labels, 't-SNE for S2 - Training Data', output_path)
        print("[INFO] Applying t-SNE for S2 for test set...")
        output_path = os.path.join(params.path_output, 't-SNE_test.png')
        apply_tsne_and_plot(test_tensor_combined, test_labels, 't-SNE for S2 - Test Data', output_path)
        
    # Interpolate (if wanted) and Normalize the train, val and test data (separate normalization if using S2 and Planet)
    if not params.use_multisensors:
        # Interpolation of the time series
        if params.interpolate:
            print("[INFO] Interpolating time series...")
            train_tensor = interpolate_and_smooth_tensor(train_tensor, params.use_data)
            val_tensor = interpolate_and_smooth_tensor(val_tensor, params.use_data)
            test_tensor = interpolate_and_smooth_tensor(test_tensor, params.use_data)
            print("Dimensions of the training tensor after interpolation:", train_tensor.shape)
            print("Dimensions of the validation tensor after interpolation:", val_tensor.shape)
            print("Dimensions of the test tensor after interpolation:", test_tensor.shape)
        
        # Normalization
        print("\n[INFO] Normalizing features...")
        train_norm, val_norm, test_norm = global_normalize(train_tensor, val_tensor, test_tensor, params.type_norm)
        train_val_dataset = np.concatenate((train_norm, val_norm), axis=0)
        print(f"[INFO] Processed train_val_dataset dimensions: {train_val_dataset.shape}")
       
    # Multisensors
    else:
        if params.interpolate:
            ### S2 interpolation
            print("[INFO] Interpolating time series for S2...")
            train_tensor_s2 = interpolate_and_smooth_tensor(train_tensor_s2, 'S2')
            val_tensor_s2 = interpolate_and_smooth_tensor(val_tensor_s2, 'S2')
            test_tensor_s2 = interpolate_and_smooth_tensor(test_tensor_s2, 'S2')
            print("Dimensions of the training tensor after interpolation:", train_tensor_s2.shape)
            print("Dimensions of the validation tensor after interpolation:", val_tensor_s2.shape)
            print("Dimensions of the test tensor after interpolation:", test_tensor_s2.shape)
            
            ### Planet interpolation
            print("[INFO] Interpolating time series for Planet...")
            train_tensor_planet = interpolate_and_smooth_tensor(train_tensor_planet, 'Planet')
            val_tensor_planet = interpolate_and_smooth_tensor(val_tensor_planet, 'Planet')
            test_tensor_planet = interpolate_and_smooth_tensor(test_tensor_planet, 'Planet')
            print("Dimensions of the training tensor after interpolation:", train_tensor_planet.shape)
            print("Dimensions of the validation tensor after interpolation:", val_tensor_planet.shape)
            print("Dimensions of the test tensor after interpolation:", test_tensor_planet.shape)
        
        # Normalization
        print("\n[INFO] Normalizing features...")
        print("Normalizing S2 features...")
        train_norm_s2, val_norm_s2, test_norm_s2 = global_normalize(train_tensor_s2, val_tensor_s2, test_tensor_s2, params.type_norm)
        train_val_s2 = np.concatenate((train_norm_s2, val_norm_s2), axis=0)
        print(f"[INFO] Processed train_val_s2 dimensions: {train_val_s2.shape}")
        
        print("Normalizing Planet features...")
        train_norm_planet, val_norm_planet, test_norm_planet = global_normalize(train_tensor_planet, val_tensor_planet, test_tensor_planet, params.type_norm)
        train_val_planet = np.concatenate((train_norm_planet, val_norm_planet), axis=0)
        print(f"[INFO] Processed train_val_planet dimensions: {train_val_planet.shape}")
        
    print("[INFO] Features normalized.")
    print("[INFO] Time: ", (time.time()-start)/60, "minutes. \n")
     
    # Compute the class frequencies and weights for the loss function
    class_counts = np.bincount(train_labels)
    
    # Compute the weights for each class (Inverse of the frequencies)
    # classes with fewer samples will have higher weights: "pay more attention to the under-represented classes"
    class_weights = 10000*(1. / class_counts)
    # class_weights = 1000000*(1. / class_counts)
    
    # Convert to PyTorch tensor
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    ### Start of the training process
    print(f"[INFO] Starting training {params.n_splits} runs ... \n")
    # Creation of a dictionnary to store the results of training for each run (accuracy, precision, recall, f1-score)
    results = {
        'accuracy': [],
        'precision': {class_name: [] for class_name in species_mapping.keys()},
        'recall': {class_name: [] for class_name in species_mapping.keys()},
        'f1-score': {class_name: [] for class_name in species_mapping.keys()},
    }
    
    train_val_dataset_n = train_val_dataset if not params.use_multisensors else train_val_s2

    ### Training the model n_splits times
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(train_val_dataset_n, train_val_labels)):
        if not params.use_multisensors:
            X_train, X_val = train_val_dataset[train_idx], train_val_dataset[val_idx]
            y_train, y_val = train_val_labels_tensor[train_idx], train_val_labels_tensor[val_idx]
            train_dataset = CustomDataset(X_train, y_train, transform=None)
            val_dataset = CustomDataset(X_val, y_val)
            test_dataset = CustomDataset(test_norm, test_labels_tensor)
        else:
            X_train_s2, X_val_s2 = train_val_s2[train_idx], train_val_s2[val_idx]            
            X_train_planet, X_val_planet = train_val_planet[train_idx], train_val_planet[val_idx]            
            y_train, y_val = train_val_labels_tensor[train_idx], train_val_labels_tensor[val_idx]
            train_dataset = CustomDualDataset(X_train_s2, X_train_planet, y_train, transform=None)
            val_dataset = CustomDualDataset(X_val_s2, X_val_planet, y_val)
            test_dataset = CustomDualDataset(test_norm_s2, test_norm_planet, test_labels_tensor)
        print("[INFO] Datasets and dataloaders ready.")
            
        # Creation of the model's architecture to apply and train with
        if params.name_model == 'InceptionTime':
            model = InceptionTime(in_channels=params.bands_depth, 
                                  number_classes=params.num_classes, 
                                  use_residual=True, 
                                  activation=nn.ReLU())
            model.to(device)
        elif params.name_model == 'DualInceptionTime':
            model = DualInceptionTime(in_channels_s2=params.bands_depth_S2, 
                                      in_channels_planet=params.bands_depth_Planet, 
                                      kernel_sizes_s2= params.kernel_sizes_s2, 
                                      kernel_sizes_planet = params.kernel_sizes_planet, 
                                      number_classes=params.num_classes)
            model.to(device)
        elif params.name_model == "HInceptionTime":
            model = HybridInceptionTime(in_channels=params.bands_depth, 
                                        kernel_sizes= params.kernel_sizes, 
                                        number_classes=params.num_classes)
            model.to(device)
        elif params.name_model == "HDualInceptionTime":
            model = DualHybridInceptionTime(in_channels_s2=params.bands_depth_S2, 
                                            in_channels_planet=params.bands_depth_Planet, 
                                            kernel_sizes_s2= params.kernel_sizes_s2, 
                                            kernel_sizes_planet = params.kernel_sizes_planet, 
                                            number_classes=params.num_classes)
            model.to(device)
        elif params.name_model == "Transformer":            
            ## Model 1
            if params.transf == 1:
                model = Transformer(d_input = params.bands_depth, 
                                    d_model = params.d_model, 
                                    d_output = params.num_classes, 
                                    q=params.q, v=params.v, h=params.h, N=params.N, 
                                    attention_size=params.attention_size,
                                    dropout=params.dropout)
                model.to(device)
            ## Model 2
            elif params.transf == 2:
                model = TSTransformerEncoderClassiregressor(feat_dim=params.bands_depth, 
                                                            d_model=params.d_model,
                                                            n_heads=params.h,
                                                            num_layers=params.N,
                                                            dim_feedforward=params.dim_feedforward,
                                                            num_classes=params.num_classes,
                                                            dropout=params.dropout,
                                                            activation=nn.ReLU(),
                                                            norm=params.normalization_layer)
                model.to(device)            
        elif params.name_model == "DualTransformer":
            ## Model 1
            if params.transf == 1:
                model = DualTransformer(d_input=params.bands_depth_Planet,
                                        d_model=params.d_model, 
                                        d_output=params.num_classes, 
                                        q=params.q, v=params.v, h=params.h, N=params.N, 
                                        attention_size=params.attention_size,
                                        dropout=params.dropout)
                model.to(device)            
            ## Model 2
            elif params.transf == 2:
                model = TSTransformerEncoderClassiregressorDual(feat_dim_s2=params.bands_depth_S2, 
                                                                feat_dim_planet=params.bands_depth_Planet, 
                                                                d_model=params.d_model,
                                                                n_heads=params.h,
                                                                num_layers=params.N, 
                                                                dim_feedforward=params.dim_feedforward,
                                                                num_classes=params.num_classes,
                                                                dropout=params.dropout, 
                                                                activation=nn.ReLU(),
                                                                norm=params.normalization_layer)
                model.to(device)
        elif params.name_model == "LITE":
            model = LITE(in_channels=params.bands_depth, 
                         kernel_sizes=params.kernel_sizes, 
                         number_classes=params.num_classes)
            model.to(device)
        elif params.name_model == "DualLITE":
            model = DualLITE(in_channels_s2=params.bands_depth_S2, 
                             in_channels_planet=params.bands_depth_Planet, 
                             kernel_sizes_s2= params.kernel_sizes_s2, 
                             kernel_sizes_planet=params.kernel_sizes_planet, 
                             number_classes=params.num_classes)
            model.to(device)

        # Creation of the training model
        train_model = TrainModel(model, train_dataset, val_dataset, test_dataset, class_weights, species_mapping)
        print(f"[INFO] Model initialized for fold {fold}.")
        
        # Training
        print(f"[INFO] Starting training for fold {fold} ...")
        report = train_model.run_n_times(fold)
        results['accuracy'].append(report['accuracy'])

        # Precision, recall, F1-score appened to their respective lists in the results dictionnary (if the class is in the species_mapping dictionnary)
        for class_name, metrics in report.items():
            if class_name in species_mapping.keys():  # Pour éviter les clés comme 'accuracy' et 'macro avg'
                results['precision'][class_name].append(metrics['precision'])
                results['recall'][class_name].append(metrics['recall'])
                results['f1-score'][class_name].append(metrics['f1-score'])
        print("[INFO] Time: ", (time.time()-start)/60, "minutes.")
        print(f"[INFO] Training completed for run {fold}. \n")
 
    # Compute the mean and standard deviation for each metric and each class
    final_results = {}
    for metric in ['precision', 'recall', 'f1-score']:
        final_results[metric] = {}
        for class_name in results[metric].keys():
            mean = np.mean(results[metric][class_name])
            std = np.std(results[metric][class_name])
            final_results[metric][class_name] = f"{mean:.4f} +/- {std:.4f}"

    accuracy_mean = np.mean(results['accuracy'])
    accuracy_std = np.std(results['accuracy'])
    final_results['accuracy'] = f"{accuracy_mean:.4f} +/- {accuracy_std:.4f}"

    with open(os.path.join(params.path_output, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # GradCAM analysis if using only 1 sensor
    # if not params.use_multisensors:
    #     print("[INFO] Starting GradCAM analysis...")
    #     if params.use_data == 'S2':
    #         formatted_dates = [datetime.strptime(date, '%Y%m%d').strftime('%d-%m') for date in params.list_dates_S2]
    #     if params.use_data == 'Planet':
    #         formatted_dates = [datetime.strptime(date, '%Y%m%d').strftime('%d-%m') for date in params.list_dates_Planet]
    #     gradcam = GradCAMAnalysis(model, target_layer=model.inception_block1.inception_1.conv_from_bottleneck_3, test_dataset=test_dataset ,device=device, formatted_dates=formatted_dates, output_folder=params.gradcam_boxplot_dir)
    #     gradcam.plot_boxplots_for_classes()
    #     print("[INFO] GradCAM analysis completed.")

    print("[INFO] Classification and vote...")
    model_paths = []
    for n in range(params.n_splits):
        model_paths.append(os.path.join(params.model_path, f'model_{n}.pth'))
    
    # Load the models from memory
    models = [test.load_model(path, model, device) for path in model_paths]

    predictions = test.predict_with_models(test_dataset, models, device, params.use_multisensors)
    test.vote_and_save(predictions, test_data, inverse_species_mapping, os.path.join(params.path_output, "output.gpkg"))
    
    print("[INFO] Classification done.")
    print("[INFO] Time: ", (time.time()-start)/60, "minutes.")
    
# __________________________________________________________________
    if (params.name_model == "Transformer" or params.name_model == "DualTransformer") and params.transf==1:
        for m in range(params.n_splits):
            model = models[m]
            get_attention_maps_by_head(model, params.N, params.h, m)
            get_attention_maps_by_class(test_dataset, model, inverse_species_mapping, m)
            # average_attention_maps_by_class = compute_average_attention_maps(attention_maps_by_class)
            # plot_attention_maps(average_attention_maps_by_class)

        print("[INFO] Time: ", (time.time()-start)/60, "minutes.")
# __________________________________________________________________

    #Stop carbon tracker
    emissions: float = tracker.stop()
    if emissions is not None:
        print(f'Carbon emission: {float(emissions)} kWh of electricity')
    else:
        print('Emission calculation failed, emissions is None')
        
if __name__ == "__main__":
    main()
