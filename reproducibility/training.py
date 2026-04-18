from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder)

from kfold_trainer import KfoldTrainer
from config import KFOLD_SPLITS

# Should be one of the RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

# Should be stead, instance, or any custom dataset defined in settings.json
DATASETS = ["BALIKESIR2025"]

# Number of epochs
NUM_EPOCHS = 20
# For all splits, train the model over defined datasets.
for train_dataset in DATASETS:
	for model_class in MODEL_CLASSES:
		for split in range(1):
                	exp_name = f"{train_dataset}_NODILATION_20EP"
                	kfold_trainer = KfoldTrainer(exp_name,model_class, train_dataset,split,epochs=NUM_EPOCHS, apply_resampling=False)
                	kfold_trainer.train()
