import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from skrebate import SURF


class FeaturesFilter():

    def __init__(self, feats_path, labels_path, save_path, tag='latest', n_feats=15, th_auc=0.5, th_vif=10, random_seed=426):
        self.feats  = pd.read_excel(feats_path, index_col=0).iloc[:,:-1]
        self.labels = pd.read_excel(labels_path, index_col=0).iloc[:,0].values
        self.save_path = save_path
        self.tag = tag
        self.n_feats = n_feats
        self.th_auc = th_auc
        self.th_vif = th_vif
        self.random_seed = random_seed

    def normalize_feats(self):
        scaler = MinMaxScaler().fit(self.feats)
        self.feats_array = scaler.transform(self.feats)

    def get_high_auc_feature(self):
        auc_list = []
        auc_idx = []
        flag = 0
        for index in range(len(self.feats_array[0])):
            feature_single = self.feats_array[:, index].reshape(-1, 1)
            auc_single = self._cacu_auc(feature_single, self.labels)
            if auc_single > self.th_auc:
                auc_list.append(auc_single)
                auc_idx.append(index)
                flag += 1
            print(' Effective / All: [{} / {}]'.format(flag, len(self.feats_array[0])))
        self.feats_array = self.feats_array[:, auc_idx]
        self.feats = self.feats.iloc[:, auc_idx]
        self.feats.to_excel(os.path.join(self.save_path, f'features_{self.tag}_auc_{self.th_auc}.xlsx'))

    def calc_vif(self):
        variables = list(range(self.feats_array.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(self.feats_array[:, variables], ix)
                for ix in range(self.feats_array[:, variables].shape[1])]
            maxloc = vif.index(max(vif))
            if max(vif) > self.th_vif:
                print('Dropping at index {} of vif : {}'.format(variables[maxloc], max(vif)))
                del variables[maxloc]
                print('The number of the rest features is {}'.format(len(variables)))
                dropped = True
        self.feats_array = self.feats_array[:, variables]
        self.feats = self.feats.iloc[:, variables]
        self.feats.to_excel(os.path.join(self.save_path, f'features_{self.tag}_vif_{self.th_vif}.xlsx'))

    def surf_filter(self):
        surf_selector = SURF(n_features_to_select=self.n_feats, n_jobs=-1)
        surf_selector.fit(self.feats_array, self.labels)
        idx_surf = surf_selector.top_features_[:self.n_feats]
        self.feats_array = self.feats_array[:, idx_surf]
        self.feats = self.feats.iloc[:, idx_surf]
        self.feats.to_excel(os.path.join(self.save_path, f'features_{self.tag}_surf_{self.n_feats}.xlsx'))

    def random_forest(self):
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        rf_clf.fit(self.feats_array, self.labels)
        importances = rf_clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        idx_rf = []
        for f in range(self.n_feats):
            idx_rf.append(indices[f])
        self.feats_array = self.feats_array[:, idx_rf]
        self.feats = self.feats.iloc[:, idx_rf]
        self.feats.to_excel(os.path.join(self.save_path, f'features_{self.tag}_rf_{self.n_feats}.xlsx'))

    def extra_trees(self):
        et_clf = ExtraTreesClassifier(n_estimators=100, random_state=self.random_seed)
        et_clf.fit(self.feats_array, self.labels)
        importances = et_clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        idx_et = []
        for f in range(self.n_feats):
            idx_et.append(indices[f])
        self.feats_array = self.feats_array[:, idx_et]
        self.feats = self.feats.iloc[:, idx_et]
        self.feats.to_excel(os.path.join(self.save_path, f'features_{self.tag}_et_{self.n_feats}.xlsx'))

    @staticmethod
    def _cacu_auc(features, label):
        cv_single = RepeatedStratifiedKFold(n_splits=5, n_repeats=50)
        clf = SVC(probability=True, gamma='auto')
        aucs = []
        for train, test in cv_single.split(features, label):
            probas_ = clf.fit(features[train], label[train]).predict_proba(features[test])
            # auc_t = roc_auc_score(label[test], probas_[:,1], multi_class='ovr', average="macro")
            auc_t = roc_auc_score(label[test], probas_[:,1])
            aucs.append(auc_t)
        auc_avg = sum(aucs)/len(aucs)
        return auc_avg


if __name__ == '__main__':

    FEATS_PATH = '/home/acy/data/lung/radiomics/feats_1012.xlsx'
    LABELS_PATN = '/home/acy/data/lung/radiomics/labels_1012.xlsx'
    SAVE_PATH = 'radiomics'
    TAG = 'new_1012'
    n_feats = 10
    th_auc = 0.5
    th_vif = 5
    random_seed = 42

    ff = FeaturesFilter(
        feats_path=FEATS_PATH,
        labels_path=LABELS_PATN,
        save_path=SAVE_PATH,
        n_feats=n_feats,
        th_auc=th_auc,
        th_vif=th_vif,
        random_seed=random_seed,
        tag=TAG
    )

    ff.normalize_feats()
    ff.get_high_auc_feature()
    ff.calc_vif()
    ff.random_forest()


    