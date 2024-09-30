# test.py
import torch
import numpy as np
from torch.utils.data import DataLoader


def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_with_models(test_dataset, models, device, use_multisensors=False):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predictions = []

    for data in test_loader:
        if use_multisensors:
            (inputs_s2, inputs_planet), _ = data
            inputs_s2 = inputs_s2.to(device)
            inputs_planet = inputs_planet.to(device)
        else:
            inputs, _ = data
            inputs = inputs.to(device)

        # Stock the predictions of each model for this sample
        sample_predictions = []
        for model in models:
            with torch.no_grad():
                if use_multisensors:
                    outputs = model(inputs_s2, inputs_planet)
                else:
                    outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                sample_predictions.append(predicted.item())
    
        # Add the list of predictions for this sample to the global list
        predictions.append(sample_predictions)        
    return np.array(predictions)


def vote_and_save(predictions, test_gdf, inverse_species_mapping, output_gpkg_path):
    # Majoritary vote to determine the final prediction
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(inverse_species_mapping)).argmax(), 1, predictions)
    
    # Convert indices to species names using the inverse dictionary
    predicted_species = [inverse_species_mapping[pred] for pred in final_predictions]

    # Compute if the prediction is correct
    correct_predictions = (predicted_species == test_gdf['esse_tri']).astype(int)
    
    # Add the results to the GeoDataFrame
    test_gdf['predicted_species'] = predicted_species
    test_gdf['correct_prediction'] = correct_predictions
    
    columns_to_keep = ['geometry', 'predicted_species', 'correct_prediction', 'esse_tri'] 

    # Delete other columns
    test_gdf = test_gdf[columns_to_keep]

    # Save the cleaned GeoDataFrame to a GeoPackage file
    test_gdf.to_file(output_gpkg_path, driver="GPKG")

