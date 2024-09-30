# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import seaborn as sns
import params 


class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, class_weights, species_mapping):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.class_weights = class_weights
        self.species_mapping = species_mapping
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_accuracy_list = []

    def prepare_data(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=params.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=params.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=params.batch_size)

    def define_model(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    def train(self):
        for epoch in range(params.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            if not params.use_multisensors:
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
            else:
                for (inputs_s2, inputs_planet), labels in self.train_loader:
                    inputs_s2, inputs_planet, labels = inputs_s2.to(self.device), inputs_planet.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs_s2, inputs_planet)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * inputs_s2.size(0)
            
            train_loss /= len(self.train_loader.dataset)
            self.train_loss_list.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                if not params.use_multisensors:
                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_accuracy += (predicted == labels.max(1)[1]).sum().item()
                else:
                    for (inputs_s2, inputs_planet), labels in self.val_loader:
                        inputs_s2, inputs_planet, labels = inputs_s2.to(self.device), inputs_planet.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs_s2, inputs_planet)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item() * inputs_s2.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_accuracy += (predicted == labels.max(1)[1]).sum().item()
            
            val_loss /= len(self.val_loader.dataset)
            self.val_loss_list.append(val_loss)
            self.scheduler.step(val_loss)

            val_accuracy /= len(self.val_loader.dataset)
            self.val_accuracy_list.append(val_accuracy)

            print(f"Epoch {epoch+1}/{params.num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    def evaluate(self, i=1):
        # Gather all true labels and predictions
        all_labels = []
        all_predictions = []

        # Evaluate on test set
        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            if not params.use_multisensors:
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    # Move predictions and labels to CPU and convert them to numpy arrays
                    all_labels.extend(labels.max(1)[1].cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            else:
                for (inputs_s2, inputs_planet), labels in self.test_loader:
                    inputs_s2, inputs_planet, labels = inputs_s2.to(self.device), inputs_planet.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs_s2, inputs_planet)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item() * inputs_s2.size(0)  # Assuming inputs_s2 and inputs_planet have the same batch size
                    _, predicted = torch.max(outputs, 1)
                    # Move predictions and labels to CPU and convert them to numpy arrays
                    all_labels.extend(labels.max(1)[1].cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

        test_loss /= len(self.test_loader.dataset)
        # Computing accuracy: Since labels are not averaged, divide by the number of labels instead of inputs
        # test_accuracy /= len(self.test_loader.dataset)
        test_accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Convert lists to numpy arrays for scikit-learn functions
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Compute classification report and confusion matrix
        report_dict = classification_report(all_labels, all_predictions, target_names=self.species_mapping.keys(), digits=4, output_dict=True)
        report = classification_report(all_labels, all_predictions, target_names=self.species_mapping.keys(), digits=4)        
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        conf_matrix2 = confusion_matrix(all_labels, all_predictions, normalize='true')

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
        # print("Confusion Matrix normalized:\n", conf_matrix2)
        
        np.save(os.path.join(params.path_output, f'conf_matrix_{i}.npy'), conf_matrix)
        np.save(os.path.join(params.path_output, f'conf_matrix_{i}_norm.npy'), conf_matrix2)
        
        with open(os.path.join(params.path_output, f'classification_report_{i}.txt'), 'w') as f:
            f.write(report)
            
        # Creation and saving of the figure for the confusion matrix
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)  # font size
        ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(self.species_mapping.keys()), yticklabels=list(self.species_mapping.keys()))
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotation of the labels on the x-axis for better readability
        # plt.xticks(rotation=45, ha="right")
        # plt.yticks(rotation=0)
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(params.path_output, f'confusion_matrix_{i}blue.png'))
        
        # Creation and saving of the figure for the confusion matrix
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)  # font size
        ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="plasma", xticklabels=list(self.species_mapping.keys()), yticklabels=list(self.species_mapping.keys()))
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotation of the labels on the x-axis for better readability
        # plt.xticks(rotation=45, ha="right")
        # plt.yticks(rotation=0)
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(params.path_output, f'confusion_matrix_{i}plasma.png'))       
        
        # Creation and saving of the figure for the confusion matrix
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)  # font size
        ax = sns.heatmap(100*conf_matrix2, annot=True, fmt=".0f", cmap="Blues", xticklabels=list(self.species_mapping.keys()), yticklabels=list(self.species_mapping.keys()))
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotation of the labels on the x-axis for better readability
        # plt.xticks(rotation=45, ha="right")
        # plt.yticks(rotation=0)
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(params.path_output, f'confusion_matrix_{i}blue_norm.png'))

        # Creation and saving of the figure for the confusion matrix
        plt.figure(figsize=(15, 10))
        sns.set(font_scale=1)  # font size
        ax = sns.heatmap(100*conf_matrix2, annot=True, fmt=".0f", cmap="plasma", xticklabels=list(self.species_mapping.keys()), yticklabels=list(self.species_mapping.keys()))
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotation of the labels on the x-axis for better readability
        # plt.xticks(rotation=45, ha="right")
        # plt.yticks(rotation=0)
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(params.path_output, f'confusion_matrix_{i}plasma_norm.png'))
        
        return report_dict

    def save_model(self, run):
        torch.save(self.model.state_dict(), os.path.join(params.model_path, f'model_{run}.pth'))
    
    def draw_loss(self, run):
        if os.path.exists(os.path.join(params.path_output, f'loss_plot_{run}.png')):
            os.remove(os.path.join(params.path_output, f'loss_plot_{run}.png'))

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_list, 'g-', label='Training Loss')
        plt.plot(self.val_loss_list, 'b-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(params.path_output, f'loss_plot_{run}.png'))

    def run_n_times(self, run):
        self.prepare_data()
        self.define_model()
        if params.city == 'Strasbourg':
            self.train()
        elif params.city == 'Nancy' and params.type != 'inf':
            self.train()
        report = self.evaluate(run)            
        self.save_model(run)
        if params.city == 'Strasbourg':
            self.draw_loss(run)
        elif params.city == 'Nancy' and params.type != 'inf':
            self.draw_loss(run)
        return report
    