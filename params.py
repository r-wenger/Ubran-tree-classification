# params.py
# _______________________________________________________________________________________
### Project parameters configuration - to check before running the code

# Area of interest
city = 'Nancy' # 'Strasbourg', 'Nancy'

# For Nancy
type = 'all' # 'inf', 'ft', 'all'

# Hyperparameters
name_model = 'HDualInceptionTime' #InceptionTime, DualInceptionTime, HInceptionTime, HDualInceptionTime, Transformer, DualTransformer, LITE, DualLITE
batch_size = 128
num_epochs = 30 # 60, 30
learning_rate = 0.001

n_splits = 5
tsne= False

# Choice of the dataset (Sentinel-2 ou Planet)
use_data = 'S2-Planet' # 'S2', 'Planet', 'S2-Planet'

# Kernel sizes
k = 22
# k=1: [2, 4, 8]
# k=2: [10, 20, 40]
# multisensor -> order: S2 then Planet
# k=11: [2, 4, 8], [2, 4, 8]
# k=12: [2, 4, 8], [10, 20, 40]
# k=21: [10, 20, 40], [2, 4, 8]
# k=22: [10, 20, 40], [10, 20, 40]

# Transformer model: 1 or 2, and parameters
transf = 1
# for both
d_model = 64 # 64 // 128 // 256  # Lattent dim
attention_size = 40 # 12 // 8 // 16 # Attention window size
dropout = 0.1 # 0.1 // 0.2  # Dropout rate
h = 4  # 8 // 4 Number of heads
N = 4  # 4 // 6 Number of encoder and decoder to stack

# for model 1
q = d_model//8  # 64/8 = 8 // 128/8 = 16 // 256/8 = 32 Query size
v = d_model//8  # 64/8 = 8 // 128/8 = 16 // 256/8 = 32 Value size

# for model 2
dim_feedforward = 128
normalization_layer = 'BatchNorm' # 'BatchNorm', 'LayerNorm'


# Interpolate or not the data
interpolate = False

# Number of species wanted
num_species = 19

type_norm = 'min_max' # 'min_max', 'mean_std'

# _______________________________________________________________________________________

# use_multisensors or not
if use_data == 'S2' or use_data == 'Planet':
    use_multisensors = False
else:
    use_multisensors = True

# Kernel sizes
if k==12:
    kernel_sizes_s2 = [2, 4, 8]
    kernel_sizes_planet = [10, 20, 40]
elif k==1 or k==11:
    kernel_sizes_s2 = [2, 4, 8]
    kernel_sizes_planet = [2, 4, 8]
elif k==2 or k==22:
    kernel_sizes_s2 = [10, 20, 40]
    kernel_sizes_planet = [10, 20, 40]
elif k==21:
    kernel_sizes_s2 = [10, 20, 40]
    kernel_sizes_planet = [2, 4, 8]
    
# Interpolate
if interpolate:
    interpol = "-interpolated-"
else:
    interpol = "-"

# Sentinel-2 dataset configuration
list_bands_S2 = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
if city == 'Strasbourg':
    list_dates_S2 = ['20220213', '20220223', '20220228', '20220305', '20220310', '20220320', '20220325', '20220504', '20220509', '20220514', '20220613', '20220618', '20220623', '20220703', '20220708', '20220713', '20220728', '20220807', '20220812', '20220822', '20220901', '20220921']
elif city == 'Nancy':
    list_dates_S2 = ['20220112', '20220226', '20220303', '20220308', '20220323', '20220328', '20220417', '20220422', '20220507', '20220517', '20220601', '20220606', '20220611', '20220616', '20220706', '20220711', '20220716', '20220731', '20220810', '20220820', '20220825', '20220830', '20220904', '20221004', '20221009', '20221024', '20221113', '20221118', '20221123']

# Planet dataset configuration
list_bands_Planet = ['B01', 'B02', 'B03', 'B04']
if city == 'Strasbourg':
    list_dates_Planet = ['20220119', '20220307', '20220308', '20220311', '20220321', '20220322', '20220323', '20220328', '20220417', '20220418', '20220419', '20220422', '20220428', '20220515', '20220518', '20220530', '20220610', '20220615', '20220617', '20220618', '20220621', '20220705', '20220713', '20220715', '20220717', '20220718', '20220803', '20220804', '20220806', '20220807', '20220808', '20220809', '20220810', '20220811', '20220812', '20220813', '20220815', '20220821', '20220822', '20220823', '20220904', '20220905', '20220921', '20220922', '20220923', '20221005', '20221009', '20221010', '20221017', '20221022', '20221027', '20221123', '20221231']
elif city == 'Nancy':
    list_dates_Planet = ['20220124', '20220209', '20220308', '20220321', '20220323', '20220324', '20220328', '20220422', '20220508', '20220513', '20220515', '20220618', '20220708', '20220710', '20220711', '20220717', '20220724', '20220803', '20220804', '20220808', '20220809', '20220810', '20220822', '20220825', '20220829', '20220831', '20220901', '20220912', '20221009', '20221024', '20221027', '20221113', '20221123']

if use_data == 'S2':
    bands_depth=len(list_bands_S2)
    temp_depth=len(list_dates_S2)
    kernel_sizes = kernel_sizes_s2
elif use_data == 'Planet':
    bands_depth=len(list_bands_Planet)
    temp_depth=len(list_dates_Planet)
    kernel_sizes = kernel_sizes_planet
elif use_multisensors:
    bands_depth_S2=len(list_bands_S2)
    temp_depth_S2=len(list_dates_S2)
    bands_depth_Planet=len(list_bands_Planet)
    temp_depth_Planet=len(list_dates_Planet)

# Tree species configuration for classification
selected_species_10 = [
    'Platanus x acerifolia', 'Acer pseudoplatanus', 'Tilia x euchlora',
    'Acer platanoides', 'Tilia cordata', 'Fraxinus excelsior',
    'Aesculus hippocastanum', 'Carpinus betulus', 'Prunus avium',
    'Acer campestre'
]
selected_species_15 = [
    'Platanus x acerifolia', 'Acer pseudoplatanus', 'Tilia x euchlora',
    'Acer platanoides', 'Tilia cordata', 'Fraxinus excelsior',
    'Aesculus hippocastanum', 'Carpinus betulus', 'Prunus avium',
    'Acer campestre', 'Robinia pseudoacacia', 'Betula pendula',
    'Pyrus calleryana', 'Populus nigra', 'Alnus glutinosa'
]
selected_species_20 = [
    'Acer campestre',  'Acer platanoides',  'Acer pseudoplatanus',  'Aesculus hippocastanum',  
    'Alnus glutinosa',  'Alnus x spaethii',  'Betula pendula',  'Carpinus betulus',  
    'Fraxinus excelsior',  'Pinus nigra',  'Platanus x acerifolia',  'Populus nigra',  
    'Prunus avium',  'Pyrus calleryana',  'Quercus robur',  'Robinia pseudoacacia',  
    'Styphnolobium japonicum',  'Taxus baccata', 'Tilia cordata', 'Tilia x euchlora'
]
selected_species_19 = [
    'Acer campestre',  'Acer platanoides',  'Acer pseudoplatanus',  'Aesculus hippocastanum',  
    'Alnus glutinosa',  'Alnus x spaethii',  'Betula pendula',  'Carpinus betulus',  
    'Fraxinus excelsior',  'Pinus nigra',  'Platanus x acerifolia',  'Populus nigra',  
    'Prunus avium',  'Pyrus calleryana',  'Quercus robur',  'Robinia pseudoacacia',  
    'Taxus baccata', 'Tilia cordata', 'Tilia x euchlora'
]
selected_species_3 = ['Platanus x acerifolia', 'Acer pseudoplatanus', 'Fraxinus excelsior']

if num_species==20:
    selected_species = selected_species_20
elif num_species==19:
    selected_species = selected_species_19
elif num_species==15:
    selected_species = selected_species_15
elif num_species==10:
    selected_species = selected_species_10

num_classes = len(selected_species)

### Temporal normalisation configuration
temporal_sampling = False

# Scenar
if name_model == "Transformer" or name_model == "DualTransformer":
    name_scenar=f"{name_model}-{transf}-{num_classes}species-{use_data}{interpol}{n_splits}times_{city}-d{d_model}-att{attention_size}-N{N}-h{h}2"
else:
    name_scenar=f"{name_model}-{num_classes}species-{use_data}-k{k}{interpol}{n_splits}times_{city}_all"

### Paths to files and folders
if city == 'Strasbourg':
    shapefile_path = "/home2020/home/geo/mlatil/data/PA15_tamp.gpkg"
elif city == 'Nancy':
    shapefile_path = "/home2020/home/geo/mlatil/data/PA15_tamp_Nancy2.gpkg"

common_path = f'/home2020/home/geo/mlatil/Results/Results_final_nancy/'
path_output = f'{common_path}results/{name_scenar}'
output_filename = f"{common_path}SITS_UrbanTrees_{name_scenar}.out"
# gradcam_boxplot_dir = f"{common_path}results/GradCAM_boxplots/{name_scenar}"
model_path = f'{common_path}models/model-{name_scenar}'

# Scenar Strasbourg
common_path_stras = f'/home2020/home/geo/mlatil/Results/Results_final_nancy/'
name_scenar_stras=f"{name_model}-{num_classes}species-{use_data}-k{k}{interpol}{n_splits}times_Strasbourg"
# name_scenar_stras=f"{name_model}-k{k}-{num_classes}species-{use_data}{interpol}{n_splits}times_mean_norm_Strasbourg"
model_path_stras = f'{common_path_stras}models/model-{name_scenar_stras}'

# Other specific configurations for the project
libelle = "esse_tri"