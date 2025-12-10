from recovar import RepresentationLearningMultipleAutoencoder
from recovar import ClassifierMultipleAutoencoder
from kfold_tester import KFoldTester
from evaluator import Evaluator, CropOffsetFilter
from sklearn.metrics import auc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

REPRESENTATION_LEARNING_MODEL_CLASS = RepresentationLearningMultipleAutoencoder
CLASSIFIER_MODEL_CLASS = ClassifierMultipleAutoencoder

SPLIT = 0
rows = []

def _eval_cross_testing(train_dataset, test_dataset, df_path):
    rows = []
    filters = [CropOffsetFilter()]

    evaluator = Evaluator(exp_name = f"SILIVRI2019_DYNAMIC_3",
                            representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS,
                            classifier_model_class = CLASSIFIER_MODEL_CLASS,
                            train_dataset = train_dataset,
                            test_dataset = test_dataset,
                            filters = filters,
                            split = SPLIT,
                            apply_resampling=False,
                            report_best_val_score_epoch=True,
                            method_params={})

    roc_vectors = evaluator.get_roc_vectors()
    roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])

    rows.append({"train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "roc_auc": roc_auc})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(df_path, mode='a', header=not os.path.exists(df_path), index=False)

def _plot_roc(train_dataset, test_dataset, resample_eq_ratio):
    filters = [CropOffsetFilter()]
    evaluator = Evaluator(exp_name = f"exp_{train_dataset}",
                            representation_learning_model_class=REPRESENTATION_LEARNING_MODEL_CLASS,
                            classifier_model_class = CLASSIFIER_MODEL_CLASS,
                            train_dataset = train_dataset,
                            test_dataset = test_dataset,
                            filters = filters,
                            split = SPLIT,
                            report_best_val_score_epoch=True,
                            method_params={})

    roc_vectors = evaluator.get_roc_vectors()
    roc_auc = auc(roc_vectors[0]["fpr"], roc_vectors[0]["tpr"])
    print(roc_auc)

    roc_data = pd.DataFrame({
        'False Positive Rate': roc_vectors[0]["fpr"],
        'True Positive Rate': roc_vectors[0]["tpr"]
    })

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=roc_data, x='False Positive Rate', y='True Positive Rate', label='ROC Curve')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Train:{train_dataset} Test:{test_dataset} ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{train_dataset}_on_{test_dataset}_{resample_eq_ratio}_tpr-fpr.png")
'''
DATASETS = [
    "BGKT_fixed",
    "ERIK_fixed",
    "SLVT_fixed",
    "CTKS_fixed",
    "GELI_fixed",
    "GONE_fixed",
    "ISK_fixed",
    "IZI_fixed",
    "KCTX_fixed",
    "KLYT_fixed",
    "MDNY_fixed",
    "MRMT_fixed",
    "ORLT_fixed",
    "OSMT_fixed",
    "TKR_fixed",
    "UKOP_fixed",
    "YLV_fixed",
]
'''
DATASETS=['SILIVRI2019']
for dataset in DATASETS:
    _eval_cross_testing("SILIVRI2019", dataset, "/home/ege/recovar/SILIVRI2019_DYNAMIC_3_self.csv")
