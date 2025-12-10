from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="SILIVRI2019_DYNAMIC_3_002",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="SILIVRI2019",
    split=0,
    epochs=100,
    batch_multiplier=3,
    keep_top_batches=0.02, 
) 

trainer.train()
