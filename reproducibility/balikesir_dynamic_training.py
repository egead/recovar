from kfold_dynamic_trainer import KfoldDynamicTrainer
from recovar import RepresentationLearningMultipleAutoencoder

trainer = KfoldDynamicTrainer(
    exp_name="BALIKESIR_DYNAMIC_8",
    model_class=RepresentationLearningMultipleAutoencoder,
    dataset="BALIKESIR2025",
    split=0,
    epochs=40,
    final_dilation_number=8
) 

trainer.train()
