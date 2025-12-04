from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="SLVT_DYNAMIC_3",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="SLVT_fixed",
    split=0,
    epochs=100,
    batch_multiplier=3,
    keep_top_batches=0.1, 
) 

trainer.train()
