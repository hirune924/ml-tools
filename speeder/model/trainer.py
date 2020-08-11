from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score, mean_squared_error, average_precision_score

class Trainer:
    def __init__(self, model, cv):
        self.model = model
        self.cv = cv

    def train(self, train_df, test_df):
        self.model.fit(train_df)

    def run_train_cv(self) -> None:
        """
        クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # mlflow[deleted]

        scores = []
        va_idxes = []
        preds = []

        # Adversarial validation[deleted]
        # 各foldで学習を行う
        for i_fold in range(self.cv.n_splits):
            # 学習を行う
            model, va_idx, va_pred, score = self.train_fold(i_fold)

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        if self.evaluation_metric == 'log_loss':
            cv_score = log_loss(self.y_train, preds, eps=1e-15, normalize=True)
        elif self.evaluation_metric == 'mean_absolute_error':
            cv_score = mean_absolute_error(self.y_train, preds)
        elif self.evaluation_metric == 'rmse':
            cv_score = np.sqrt(mean_squared_error(self.y_train, preds))
        elif self.evaluation_metric == 'auc':
            cv_score = roc_auc_score(self.y_train, preds)
        elif self.evaluation_metric == 'prauc':
            cv_score = average_precision_score(self.y_train, preds)

        # 予測結果の保存[deleted]
        # mlflowでlogging[deleted]