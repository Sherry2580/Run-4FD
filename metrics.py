import numpy as np

class My_metrics():
    """
    設計用於存儲和計算模型的性能指標（如準確率、精確率、召回率、F1 分數、AUC 等）。
    多折驗證的指標追蹤，選取最佳模型表現。
    """
    def __init__(self, metrics=['accs', 'precisions', 'recalls', 'f1s', 'aucs', 'aprs'], folds=10):
        self.metrics = {}
        self.folds = folds
        for metric in metrics:
            self.metrics[metric] = {
                f'fold{i+1}': [] for i in range(folds)
            }

    def get_fold_best(self, metric, fold):
        return max(self.metrics[metric][fold]), self.metrics[metric][fold].index(max(self.metrics[metric][fold]))

    def get_item(self, metric, fold, idx):
        return self.metrics[metric][fold][idx]
    
    def get_final(self, metric):
        return np.mean([max(self.metrics[metric][f'fold{i+1}']) for i in range(self.folds)]), np.std([max(self.metrics[metric][f'fold{i+1}']) for i in range(self.folds)])