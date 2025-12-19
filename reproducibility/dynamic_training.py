from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="SILIVRI2019_DYNAMIC_64",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="SILIVRI2019",
    split=0,
    epochs=100,
    final_dilation_number=64
) 

trainer.train()
