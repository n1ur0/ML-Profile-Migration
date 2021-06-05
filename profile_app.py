import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.offline as py
# from plotly.offline import iplot
import plotly.express as px
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest

from plotly.graph_objs import *
from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import streamlit as st
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import pickle
import shap
import matplotlib.pyplot as plt
from os import walk
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.figure_factory as ff
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import LabelEncoder, Normalizer, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import graphviz as graphviz

import streamlit.components.v1 as components

from sklearn import tree
import pickle
# import umap.umap_ as umap
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import DBSCAN

from sklearn.manifold import TSNE

from xgboost import plot_importance, plot_tree
import xgboost as xgb
from xgboost import XGBClassifier
# import shap
from sklearn.inspection import permutation_importance


import pandas as pd
import numpy as np
from os import walk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import random
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from itertools import cycle, islice

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# precision-recall curve and f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from chart_studio.plotly import plot, iplot


# precision-recall curve and f1
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Altice Profile Migration Prediction')


@st.cache(allow_output_mutation=True)  # 👈 This function will be cached
def split_data(PATH, missing, k_fold=6):
    train_data = []
    val_data = []
    test_data = []
    labels = ['TimeCluster_lab', 'CatCluster_lab', 'Change_TimeCluster_lab', 'Change_CatCluster_lab']
    
    if missing == 'no':
        for i in range(1, k_fold+1):
            train = pd.read_csv(PATH+f'folder{i}_train.csv')
            val = pd.read_csv(PATH+f'folder{i}_valid.csv')
            test = pd.read_csv(PATH+f'folder{i}_test.csv')
            
            for label in labels:
                train = train[train[label] != 'missing']
                val = val[val[label] != 'missing']
                test = test[test[label] != 'missing']

            train_data.append(train)
            val_data.append(val)
            test_data.append(test)
            
    if missing == 'yes':
        for i in range(1, k_fold+1):
            train = pd.read_csv(PATH+f'folder{i}_train.csv')
            val = pd.read_csv(PATH+f'folder{i}_valid.csv')
            test = pd.read_csv(PATH+f'folder{i}_test.csv')

            for label in labels:
                train[label] = train[label].apply(
                    lambda x: str(-1) if x == 'missing' else x)
                val[label] = val[label].apply(
                    lambda x: str(-1) if x == 'missing' else x)
                test[label] = test[label].apply(
                    lambda x: str(-1) if x == 'missing' else x)

            train_data.append(train)
            val_data.append(val)
            test_data.append(test)
            
    return train_data, val_data, test_data


@st.cache(allow_output_mutation=True)  # 👈 This function will be cached
def load_dataframe(missing, path='data_predict/foldsv3/'):
    train_data, val_data, test_data = split_data(path, missing, k_fold=6)
    data = [train_data, val_data, test_data]
    return data

def load_models(clf_name):
    if clf_name == 'random_forest':
        return RandomForestClassifier
    
    elif clf_name == 'xgb':
        return XGBClassifier
    
    elif clf_name == 'ann':
        return MLPClassifier
    
    elif clf_name == 'dtree':
        return tree.DecisionTreeClassifier
    
    elif clf_name == 'logreg':
        return LogisticRegression
    
    elif clf_name == 'knn':
        return KNeighborsClassifier
    
    elif clf_name == 'voting':
        return VotingClassifier
    

@st.cache(allow_output_mutation=True)  # 👈 This function will be cached
def featu_imp(target_val):
    df_cluster = pd.read_csv('features_clusters.csv')
    target = target_val
    features_drop = ['device', 'TimeCluster', 'CatCluster']

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    Y = df_cluster[target]
    X = df_cluster.drop(features_drop, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=1 - train_ratio)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    X_val, X_val_test, y_val, y_val_test = train_test_split(
        X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

    dval = xgb.DMatrix(X_val, y_val)
    watchlist = [(dtrain, 'train'), (dval, 'validate')]

    n_classes = len(sorted(Y.unique()))

    # val_ratio = 0.3
    # ntree = 300
    early_stop = 100
    res = {'eval_metric': 'error'}

    params = {'colsample_bytree': 0.80,
                'learning_rate': 0.004,
                #           'max_depth': 50,
                'min_child_weight': 5,
                'n_estimators': 5000,
                'objective': 'binary:logistic',
                'reg_alpha': 1,
                'reg_lambda': 0.4,
                'scale_pos_weight': 0,
                'silent': 1,
                'subsample': 0.8,
                'booster': 'gbtree',
                'eta': 0.1,
                'seed': 42,
                'num_classes': n_classes
                }


    clf = OneVsRestClassifier(XGBClassifier(params=params, num_boost_round=2000, evals_result=res,
                                            early_stopping_rounds=early_stop, verbose_eval=False, random_state=42))

    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    y_val_pred = clf.predict(X_val_test)
    
    return y_pred, y_val_pred, y_val_test, y_val, X_val_test, y_test, X_test, X_train, X, clf

# st.text('Loading data..')

# st.text('Data in folds loaded!')



option = st.sidebar.selectbox(
    'Demo Guide',
    ['Ready to start!','Analyse Labels','Analyse Correlations', 'Prediction - Cross Validation', 'Feature Importance'], )

if option not in  ['Feature Importance', 'Ready to start!']:
    with st.spinner(text='Loading data...'):
        missing = st.radio('Include missing label?', ['no', 'yes'])
        if missing:
            data = load_dataframe(missing)
        st.success('Done loading datasets!')

if option == 'Analyse Labels':
    my_fold = st.sidebar.selectbox(
        'Fold',
        list(range(1,7)))
    
    my_label = st.sidebar.selectbox(
        'Label Name',
        list(data[0][0].head().columns[-4:]))

    st.header(f'{my_label} distribution (Fold: {my_fold})')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True, sharex=True)
    sns.countplot(ax=axes[0], data=data[0][my_fold-1], x=my_label)
    axes[0].set_title(f'Train set')
    sns.countplot(ax=axes[1], data=data[1][my_fold-1], x=my_label)
    axes[1].set_title(f'Validation set')
    sns.countplot(ax=axes[2], data=data[2][my_fold-1], x=my_label)
    axes[2].set_title(f'Test set')
        
    st.pyplot(fig)
    
if option == 'Analyse Correlations':
    
    st.dataframe(data[0][0].head())
    my_fold = st.sidebar.selectbox(
        'Fold',
        list(range(1, 7)))

    my_label = st.sidebar.selectbox(
        'Label Name',
        list(data[0][0].head().columns[-4:]))

    data_set = st.radio('What set I want to perform correlation?', ['Train','Validation', 'Test'])
    st.header(f'Correlation Heatmap (Fold: {my_fold})')

    corr = data[['Train', 'Validation', 'Test'].index(
        data_set)][my_fold-1].corr('spearman')

    trace = go.Heatmap(z=corr.values,
                    x=corr.index.values,
                    y=corr.columns.values,
                    colorscale= 'YlGnBu')

    data = [trace]
    st.plotly_chart(data)
    
    corr_pairs = corr.unstack().sort_values().drop_duplicates()
    st.header('Top correlation pairs:')
    st.dataframe(corr_pairs[abs(corr_pairs) > 0.75].sort_values(ascending=False))

    
if option == 'Feature Importance':
    target_val = st.sidebar.selectbox(
        'Target Variable:',
        ['CatCluster','TimeCluster'])
    with st.spinner(text=f'Training feature importance model with target {target_val}...'):
        y_pred, y_val_pred, y_val_test, y_val, X_val_test, y_test, X_test, X_train, X, clf = featu_imp(
            target_val)
    st.sidebar.success('Feature Importance trained')
    
    conf_matrix = st.sidebar.radio('Confusion Matrix', ['disable', 'enable'])

    if conf_matrix == 'enable':
        st.header('Confusion Matrix: Validation Set')
        cm = confusion_matrix(y_val_test, y_val_pred)
        plt.figure(figsize=(20, 8))
        sns.heatmap(data=cm, linewidths=.5, annot=True, square=True,  cmap='Blues')

        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(
            clf.score(X_val_test, y_val_test))
        plt.title(all_sample_title, size=15)

        print('XGboost Model for Cluster Evaluation')
        print('Classification report:\n',
            classification_report(y_val_test, y_val_pred))
        st.pyplot(plt)

        st.header('Confusion Matrix: Test Set')
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(20, 8))
        sns.heatmap(data=cm, linewidths=.5, annot=True, square=True,  cmap='Blues')

        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(clf.score(X_test, y_test))
        plt.title(all_sample_title, size=15)

        print('XGboost Model for Cluster Evaluation')
        print('Classification report:\n', classification_report(y_test, y_pred))
        st.pyplot(plt)

    check_fimp = st.sidebar.radio(
        'Feature Importance plots', ['disable', 'enable'])

    if check_fimp == 'enable':
        for i in clf.classes_:
            # plt.rcParams['figure.figsize'] = [10, 7]
            
            st.header(f'\nCluster {i} vs All:')
            feature_names, importance = X_train.columns.to_list(), clf.estimators_[
                int(i)-1].feature_importances_
            dic = []
            for f, imp in zip(feature_names, importance):
                dic.append([f, imp])
            df_imp = pd.DataFrame(dic, columns=['feature_name', 'feature_importance'])
            df_imp = df_imp.set_index('feature_name')
            df_imp = df_imp.sort_values('feature_importance', ascending=False)
            st.dataframe(df_imp)
        #     _ = plot_importance(clf.estimators_[int(i)], height=1.1)

        #     plt.barh(feature_names, clf.estimators_[int(i)].feature_importances_)
            sorted_idx = clf.estimators_[int(i)-1].feature_importances_.argsort()
            plt.barh(np.array(feature_names)[
                    sorted_idx][-30:], clf.estimators_[int(i)-1].feature_importances_[sorted_idx][-30:])
            plt.xlabel("Xgboost Feature Importance")
            st.pyplot()

    check_shap1 = st.sidebar.radio(
        'Shap importance plots', ['disable', 'enable'])

    if check_shap1 == 'enable':
        for i in clf.classes_:
            model = clf.estimators_[int(i)-1]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.header(f'Cluster {i} vs All')
            shap.summary_plot(shap_values, X, plot_type="bar")
            st.pyplot()

    check_shap2 = st.sidebar.radio(
        'Shap values plots', ['disable', 'enable'])

    if check_shap2 == 'enable':
        for i in clf.classes_:
            model = clf.estimators_[int(i)-1]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.header(f'Cluster {i} vs All')
            # summarize the effects of all the features
            shap.summary_plot(shap_values, X)
            st.pyplot()

elif option == 'Prediction - Cross Validation':

    clf_name = st.sidebar.selectbox(
        'Classification model:',
        ['Pick one!', 'random_forest', 'xgb', 'ann', 'dtree', 'logreg', 'knn', 'voting'])

    if clf_name != 'Pick one!':
        target = st.sidebar.selectbox(
            'Target Variable:',
            data[0][0].head().columns[-4:].insert(0, 'Pick one!'))
        
        if target != 'Pick one!' and clf_name != 'Pick one!':
            n_classes = data[0][0][target].nunique()

            cv_scores_val = []
            cv_scores_test = []
            k_fold = 6
            
            if target in ['TimeCluster_lab', 'Change_TimeCluster_lab']:
                feat_remove = ['Séries_x',
                               'Filmes_x',
                               'Documentários_x',
                               'Informaçăo_x',
                               'Entretenimento_x',
                               'Generalistas_x',
                               'Desporto_x',
                               'Infantil_x',
                               'Outros_x',
                               'Lifestyle_x',
                               'Internacionais_x',
                               'Música_x',
                               'd_cat1_x',
                               'd_cat2_x',
                               'd_cat3_x',
                               'd_cat4_x',
                               'd_cat5_x',
                               'd_cat6_x',
                               'CatCluster_x',
                               'm_cat1_x',
                               'b_cat1_x',
                               'm_cat2_x',
                               'b_cat2_x',
                               'm_cat3_x',
                               'b_cat3_x',
                               'm_cat4_x',
                               'b_cat4_x',
                               'm_cat5_x',
                               'b_cat5_x',
                               'm_cat6_x',
                               'b_cat6_x',
                               'Séries_y',
                               'Filmes_y',
                               'Documentários_y',
                               'Informaçăo_y',
                               'Entretenimento_y',
                               'Generalistas_y',
                               'Desporto_y',
                               'Infantil_y',
                               'Outros_y',
                               'Lifestyle_y',
                               'Internacionais_y',
                               'Música_y',
                               'd_cat1_y',
                               'd_cat2_y',
                               'd_cat3_y',
                               'd_cat4_y',
                               'd_cat5_y',
                               'd_cat6_y',
                               'CatCluster_y',
                               'm_cat1_y',
                               'b_cat1_y',
                               'm_cat2_y',
                               'b_cat2_y',
                               'm_cat3_y',
                               'b_cat3_y',
                               'm_cat4_y',
                               'b_cat4_y',
                               'm_cat5_y',
                               'b_cat5_y',
                               'm_cat6_y',
                               'b_cat6_y',
                               'change_CatCluster',
                               'diff1_c',
                               'diff2_c',
                               'diff3_c',
                               'diff4_c',
                               'diff5_c',
                               'diff6_c',
                               'device',
                               'CatCluster_lab',
                               'TimeCluster_lab',
                               'CatCluster_lab',
                               'Change_TimeCluster_lab',
                               'Change_CatCluster_lab']


                    
                    # feat_remove = [current_place.rstrip() for current_place in fp.readlines()]
                                    
            elif target in ['CatCluster_lab', 'Change_CatCluster_lab']:
                feat_remove = ['00:00:00_06:45:00_x',
                               '06:45:00_11:45:00_x',
                               '11:45:00_19:15:00_x',
                               '19:15:00_23:59:59_x',
                               'Weekend/Holidays_x',
                               'd1_x',
                               'd2_x',
                               'd3_x',
                               'd4_x',
                               'd5_x',
                               'd6_x',
                               'TimeCluster_x',
                               'm1_x',
                               'b1_x',
                               'm2_x',
                               'b2_x',
                               'm3_x',
                               'b3_x',
                               'm4_x',
                               'b4_x',
                               'm5_x',
                               'b5_x',
                               'm6_x',
                               'b6_x',
                               '00:00:00_06:45:00_y',
                               '06:45:00_11:45:00_y',
                               '11:45:00_19:15:00_y',
                               '19:15:00_23:59:59_y',
                               'Weekend/Holidays_y',
                               'd1_y',
                               'd2_y',
                               'd3_y',
                               'd4_y',
                               'd5_y',
                               'd6_y',
                               'TimeCluster_y',
                               'm1_y',
                               'b1_y',
                               'm2_y',
                               'b2_y',
                               'm3_y',
                               'b3_y',
                               'm4_y',
                               'b4_y',
                               'm5_y',
                               'b5_y',
                               'm6_y',
                               'b6_y',
                               'change_TimeCluster',
                               'diff1_t',
                               'diff2_t',
                               'diff3_t',
                               'diff4_t',
                               'diff5_t',
                               'diff6_t',
                               'device',
                               'CatCluster_lab',
                               'TimeCluster_lab',
                               'CatCluster_lab',
                               'Change_TimeCluster_lab',
                               'Change_CatCluster_lab']
                
            if clf_name != 'ann':
                with open(f'param_config/{clf_name}.json', 'r') as fp:
                    parameters = json.load(fp)
            else:
                parameters = {
                    "max_iter": [
                        20,
                        50,
                        100
                    ],
                    "solver": [
                        "adam"
                    ],
                    "learning_rate": [
                        "adaptive"
                    ],
                    "momentum": [
                        0.9,
                        0.75
                    ],
                    "hidden_layer_sizes": [
                        (20,),
                        (20, 30),
                        (100,),
                        (20, 50, 50, 20)
                    ],
                }
            
                
            hyper_param = st.sidebar.radio('Hyperparameter Optimization:', [
                'Grid Search', 'Random Search'])
            
            if hyper_param == 'Grid Search':
                grid_search = True
                n_rand = ''
            else:
                grid_search = False
                n_rand = int(st.sidebar.number_input(
                        'Number of random iterations:', min_value=0, max_value=30, value=0))
            
            check_smote = st.sidebar.radio('SMOTE:', [
                'disable', 'enable'])
            # 👈 This function will be cached
            @st.cache(allow_output_mutation=True)
            def feature_selection():
                # Filter selection
                
                ## Correlation Selection
                
                def cor_selector(X, y, num_feats):
                    cor_list = []
                    feature_name = X.columns.tolist()
                    # y = y.astype('int')
                    y = y.apply(lambda x: int(float(x)))
                    
                    # calculate the correlation with y for each feature
                    for i in X.columns.tolist():
                        cor = np.corrcoef(X[i], y)[0, 1]
                        cor_list.append(cor)
                    # replace NaN with 0
                    cor_list = [0 if np.isnan(i) else i for i in cor_list]
                    # feature name
                    cor_feature = X.iloc[:, np.argsort(
                        np.abs(cor_list))[-num_feats:]].columns.tolist()
                    # feature selection? 0 for not select, 1 for select
                    cor_support = [True if i in cor_feature else False for i in feature_name]
                    return cor_support, cor_feature
            
                ## Chi2

                corr_list = []
                chi_list = []
                rfe_list = []
                emb_list = []
                emb_lr_list = []
                lgbm_list = []
                

                for fold in range(k_fold):
                    X_train = data[0][fold].drop(feat_remove, axis=1)
                    y_train = data[0][fold][target]
                    if check_smote == 'enable':
                        smote = SMOTE(random_state = 101)
                        X_train, y_train = smote.fit_resample(
                        X_train, y_train)
                    
                    X_norm = MinMaxScaler().fit_transform(X_train)
                    num_feats = len(X_train.columns)
                    # num_feats = 15
                    cor_support, cor_feature = cor_selector(
                        X_train, y_train, num_feats)
                    corr_list.append(len(cor_feature))
                    
                    chi_selector = SelectKBest(chi2, k=num_feats)
                    chi_selector.fit(X_norm, y_train)
                    chi_support = chi_selector.get_support()
                    chi_feature = X_train.loc[:, chi_support].columns.tolist()
                    chi_list.append(len(chi_feature))
                    
                    rfe_selector = RFE(estimator=LogisticRegression(),
                                                      n_features_to_select=num_feats, step=10, verbose=5)
                    rfe_selector.fit(X_norm, y_train)
                    rfe_support = rfe_selector.get_support()
                    rfe_feature = X_train.loc[:, rfe_support].columns.tolist()
                    rfe_list.append(len(rfe_feature))

                    embeded_lr_selector = SelectFromModel(
                        LogisticRegression(penalty="l2"), max_features=num_feats)
                    embeded_lr_selector.fit(X_norm, y_train)

                    embeded_lr_support = embeded_lr_selector.get_support()
                    embeded_lr_feature = X_train.loc[:, embeded_lr_support].columns.tolist()
                    emb_lr_list.append(len(embeded_lr_feature))

                    embeded_rf_selector = SelectFromModel(
                        RandomForestClassifier(n_estimators=100), max_features=num_feats)
                    embeded_rf_selector.fit(X_train, y_train)

                    embeded_rf_support = embeded_rf_selector.get_support()
                    embeded_rf_feature = X_train.loc[:,
                                                    embeded_rf_support].columns.tolist()
                    emb_list.append(len(embeded_rf_feature))

                    X_lgbc = X_train.rename(
                        columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

                    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                                        reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

                    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
                    embeded_lgb_selector.fit(X_lgbc, y_train)

                    embeded_lgb_support = embeded_lgb_selector.get_support()
                    embeded_lgb_feature = X_train.loc[:,
                                                      embeded_lgb_support].columns.tolist()
                    lgbm_list.append(len(embeded_lgb_feature))


                cor_cols = [
                    f'Fold {fold} selected features' for fold in range(k_fold)]
                df_corr_select = pd.DataFrame(
                    corr_list, cor_cols, columns=['Corr Selected Features'])
                # st.dataframe(df_corr_select)
                
                chi_cols = [f'Fold {fold} selected features' for fold in range(k_fold)]
                df_chi_select = pd.DataFrame(
                    chi_list, chi_cols, columns=['Chi Selected Features'])
                # st.dataframe(df_chi_select)
                
                rfe_cols = [f'Fold {fold} selected features' for fold in range(k_fold)]
                df_rfe_select = pd.DataFrame(
                    rfe_list, rfe_cols, columns=['RFE Selected Features'])
                # st.dataframe(df_rfe_select)
                
                
                emb_lr_cols = [
                    f'Fold {fold} selected features' for fold in range(k_fold)]
                df_rfe_select = pd.DataFrame(
                    emb_lr_list, emb_lr_cols, columns=['Embeded Selected Features'])
                # st.dataframe(df_rfe_select)
                
                emb_cols = [f'Fold {fold} selected features' for fold in range(k_fold)]
                df_rfe_select = pd.DataFrame(
                    emb_list, emb_cols, columns=['Embeded Selected Features'])
                # st.dataframe(df_rfe_select)
                
                lgbm_cols = [
                    f'Fold {fold} selected features' for fold in range(k_fold)]
                df_lgbm_select = pd.DataFrame(
                    lgbm_list, lgbm_cols, columns=['Embeded Selected Features'])
                # st.dataframe(df_lgbm_select)
                
                feature_name = X_train.columns.tolist()
                # put all selection together
                feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
                # count the selected times for each feature
                feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
                # display the top 100
                feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
                feature_selection_df.index = range(1, len(feature_selection_df)+1)
                # st.write(
                    # feature_selection_df[feature_selection_df.Total >= 5]['Feature'].values)
                
                return feature_selection_df[feature_selection_df.Total >=
                                            5]['Feature'].tolist(), feature_selection_df
            
            check_feat_select = st.sidebar.radio('Feature Selection:', [
                'disable', 'enable'])
            check_tree_importance = None
            if clf_name in ['dtree', 'random_forest', 'xgb']:
                check_tree_importance = st.sidebar.radio('Feature Importance Tree Based Models:', [
                    'disable', 'enable'])
            
            if check_feat_select == 'enable':
                f_select, feature_selection_df = feature_selection()
                X_train = data[0][0].drop(feat_remove, axis=1)
                num_feats = len(X_train.columns)
                st.header(f'Best Features Selected towards the target variable ({target})')

                st.dataframe(feature_selection_df.head(num_feats))


            
            @st.cache(allow_output_mutation=True)  # 👈 This function will be cached
            def get_model_results(parameters, clf_name, grid_search=True):
                best_param_fold = {}
                reports = {}
                saved_models = []
                random_param = {}
                grid = list(ParameterGrid(parameters))

                for fold in range(k_fold):
                    if check_feat_select == 'enable':
                        X_train = data[0][fold][f_select]
                        y_train = data[0][fold][target]
                        
                        if check_smote == 'enable': 
                            smote = SMOTE(random_state = 101)
                            X_train, y_train = smote.fit_resample(
                                X_train, y_train)

                        X_val = data[1][fold][f_select]
                        y_val = data[1][fold][target]

                        X_test = data[2][fold][f_select]
                        y_test = data[2][fold][target]
                    
                    else:
                        X_train = data[0][fold].drop(
                                            feat_remove, axis=1)
                        y_train = data[0][fold][target]

                        X_val = data[1][fold].drop(feat_remove, axis=1)
                        y_val = data[1][fold][target]

                        X_test = data[2][fold].drop(feat_remove, axis=1)
                        y_test = data[2][fold][target]    
                
                    if not grid_search:
                        for _ in range(n_rand):
                            for k, v in parameters.items():
                                random_param[k] = random.choice(v)
                            grid_param = random_param
                            if clf_name == 'xgb':
                                clf = load_models(clf_name)(params=grid_param, random_state=42)
                                
                            elif clf_name == 'dtree':
                                with open(f'prediction_objs/dtree_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_dtree, _, _, _, _ = comp
                                model1 = load_models('dtree')(**best_param_fold_dtree[k_fold],
                                                              random_state=42)

                                with open(f'prediction_objs/logreg_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_logreg, _, _, _, _ = comp
                                model2 = load_models('logreg')(**best_param_fold_logreg[k_fold],
                                                               random_state=42)

                                with open(f'prediction_objs/ann_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_ann, _, _, _, _ = comp
                                model3 = load_models('ann')(**best_param_fold_ann[k_fold],
                                                            random_state=42)

                                with open(f'prediction_objs/knn_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_knn, _, _, _, _ = comp

                                model4 = load_models('knn')(
                                    **best_param_fold_knn[k_fold])

                                clf = load_models(clf_name)(
                                    estimators=[('dt', model1), ('lr', model2), ('ann', model3), ('knn', model4)], **grid_param)
                                
                            else:
                                clf = load_models(clf_name)(
                                    **grid_param)
                                
                            clf.fit(X_train, y_train)

                            pred_val = clf.predict(X_val)
                            score_val = metrics.f1_score(
                                y_val.values, pred_val, average='macro')

                            cv_scores_val.append(score_val)

                            if score_val >= max(cv_scores_val):
                                best_param = grid_param
                                
                    elif grid_search:
                        for grid_param in grid:
                            if clf_name == 'xgb':
                                clf = load_models(clf_name)(params=grid_param, random_state=42)
                                
                            elif clf_name == 'voting':
                                with open(f'prediction_objs/dtree_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_dtree, _, _, _, _ = comp
                                model1 = load_models('dtree')(**best_param_fold_dtree[k_fold],
                                                            random_state=42)

                                with open(f'prediction_objs/logreg_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_logreg, _, _, _, _ = comp
                                model2 = load_models('logreg')(**best_param_fold_logreg[k_fold],
                                                            random_state=42)

                                with open(f'prediction_objs/ann_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_ann, _, _, _, _ = comp
                                model3 = load_models('ann')(**best_param_fold_ann[k_fold],
                                                            random_state=42)

                                with open(f'prediction_objs/knn_{target}_{hyper_param}{n_rand}_missing_{missing}_objs', 'rb') as fp:
                                    comp = pickle.load(fp)
                                    best_param_fold_knn, _, _, _, _ = comp

                                model4 = load_models('knn')(
                                    **best_param_fold_knn[k_fold])

                                clf = load_models(clf_name)(
                                    estimators=[('dt', model1), ('lr', model2), ('ann', model3), ('knn', model4)], **grid_param)

                            else:
                                clf = load_models(clf_name)(
                                    **grid_param)
                                
                            clf.fit(X_train, y_train)

                            pred_val = clf.predict(X_val)
                            score_val = metrics.f1_score(y_val.values, pred_val, average='macro')

                            cv_scores_val.append(score_val)

                            if score_val >= max(cv_scores_val):
                                best_param = grid_param
                        

                    best_param_fold[fold+1] = best_param
                    
                    if clf_name == 'xgb':
                        clf = load_models(clf_name)(params=best_param_fold[fold+1], random_state=42)
                        
                    elif clf_name == 'voting':
                        with open(f'prediction_objs/dtree_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'rb') as fp:
                            comp = pickle.load(fp)
                            best_param_fold_dtree, _, _, _, _ = comp
                        model1 = load_models('dtree')(**best_param_fold_dtree[k_fold],
                            random_state=42)
                        
                        with open(f'prediction_objs/logreg_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'rb') as fp:
                            comp = pickle.load(fp)
                            best_param_fold_logreg, _, _, _, _ = comp
                        
                        model2 = load_models('logreg')(**best_param_fold_logreg[k_fold],
                            random_state=42)
                        
                        with open(f'prediction_objs/ann_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'rb') as fp:
                            comp = pickle.load(fp)
                            best_param_fold_ann, _, _, _, _ = comp
                        model3 = load_models('ann')(**best_param_fold_ann[k_fold],
                            random_state=42)
                        
                        with open(f'prediction_objs/knn_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'rb') as fp:
                            comp = pickle.load(fp)
                            best_param_fold_knn, _, _, _, _ = comp
                            
                        model4 = load_models('knn')(**best_param_fold_knn[k_fold])

                        clf = load_models(clf_name)(
                            estimators=[('dt', model1), ('lr', model2), ('ann', model3), ('knn', model4)], **best_param_fold[fold+1])
                    else:
                        clf = load_models(clf_name)(
                            **best_param_fold[fold+1])
                    
                    clf.fit(X_train, y_train)
                    pred_test = clf.predict(X_test)
                    score_test = metrics.f1_score(
                        y_test.values, pred_test, average='macro')
                    report = classification_report(y_test.values, pred_test)

                    reports[fold+1] = report
                    cv_scores_test.append(score_test)
                    saved_models.append(clf)

                return best_param_fold, reports, cv_scores_val, cv_scores_test, saved_models
            
            # run = st.sidebar.button('Run Model')
            _, _, filenames = next(walk('prediction_objs/'))
            # if run:
            if grid_search or (n_rand > 2 and not grid_search):
                if f'{clf_name}_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs' not in filenames:
                    best_param_fold, reports, cv_scores_val, cv_scores_test, saved_models = get_model_results(
                        parameters, clf_name, grid_search=grid_search)
                    comp = best_param_fold, reports, cv_scores_val, cv_scores_test, saved_models
                    
                    with open(f'prediction_objs/{clf_name}_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'wb') as fp:
                        pickle.dump(comp, fp)
                else:
                    st.sidebar.success('Already Trained')

                    with open(f'prediction_objs/{clf_name}_{target}_{hyper_param}{n_rand}_missing_{missing}_feature_selection_{check_feat_select}_SMOTE_{check_smote}_objs', 'rb') as fp:
                        comp = pickle.load(fp)
                    best_param_fold, reports, cv_scores_val, cv_scores_test, saved_models = comp


                show_param = st.radio('Display all hyper_parameters.', ['disable', 'enable'])
                if show_param == 'enable':
                    st.json(parameters)
                else:
                    st.empty()
                st.header(f'{clf_name}: best parameters for test set:')
                st.table(best_param_fold)
                
                cv_scores_mean_val = []
                ratio = int(len(cv_scores_val)/len(cv_scores_test))
                for i in range(len(cv_scores_val)+1):
                    if i % ratio == 0 and i != 0:
                        res = np.mean(cv_scores_val[i-ratio:i])
                        cv_scores_mean_val.append(res)

                plt.xticks()
                ax = pd.DataFrame(cv_scores_mean_val, index=list(range(1, 7))).plot(figsize=(15, 8))
                pd.DataFrame(cv_scores_test, index=list(range(1, 7))).plot(ax=ax)
                ax.legend(['Validation', 'Test'])
                plt.xlabel('Folds')
                plt.ylabel('F1-Score')
                st.pyplot(plt)

                check_whitebox = st.sidebar.radio(
                    'White box models', ['disable', 'enable'])

                
                if len(reports) > 0:
                    my_fold = st.slider('Fold Slider', min_value=1, max_value=6)
                    

                    if check_tree_importance == 'enable':
                        st.header('Tree Based Feature Importance')
                        
                        if check_feat_select == 'disable':
                            feat_importances = pd.Series(
                                saved_models[my_fold-1].feature_importances_, index=data[0][my_fold-1].drop(
                                    feat_remove, axis=1).columns)
                            
                            fig7 = px.bar(feat_importances.nlargest(10), y=feat_importances.nlargest(10).index, x=0,
                                          orientation='h')
                            st.plotly_chart(fig7)
                            
                        else:
                            feat_importances = pd.Series(
                                saved_models[my_fold-1].feature_importances_, index=data[0][my_fold-1][f_select].columns)
                            
                            fig8 = px.bar(feat_importances.nlargest(10), y=feat_importances.nlargest(10).index, x=0,
                                          orientation='h')
                            st.plotly_chart(fig8)
                    
                    if check_whitebox == 'enable':
                        st.header('"White-box" model')
                        if clf_name == 'dtree':
                            st.subheader('Explained Decision Tree')
                            
                            my_class = data[0][my_fold-1][target].unique()
                            
                            # if target in ['Change_TimeCluster_lab', 'Change_CatCluster_lab']:
                            #     my_class = ['no', 'yes']
                                
                            # else:
                            #     my_class = data[0][my_fold-1][target].unique()

                            if check_feat_select == 'disable':
                                dot_data = tree.export_graphviz(saved_models[my_fold-1], out_file=None,
                                                                feature_names=data[0][my_fold-1].drop(
                                                                    feat_remove, axis=1).columns,
                                                                class_names=my_class,
                                                                filled=True, rounded=True,
                                                                special_characters=True)
                                
                            else:
                                dot_data = tree.export_graphviz(saved_models[my_fold-1], out_file=None,
                                                                feature_names=data[0][my_fold-1][f_select].columns,
                                                                class_names=my_class,
                                                                filled=True, rounded=True,
                                                                special_characters=True)

                                

                            st.graphviz_chart(dot_data, use_container_width=True)

                        elif clf_name == 'logreg':
                            st.subheader('Explained Logistic Regression Coefficients')

                            if target in ['Change_TimeCluster_lab', 'Change_CatCluster_lab']:
                                ind_coef = 0
                            
                            else:
                                ind_coef = st.selectbox(
                                    'Label', range(n_classes))
                            
                            log_odds = np.exp(
                                saved_models[my_fold-1].coef_[ind_coef])
                            
                            if check_feat_select == 'disable':
                                df_coef = pd.DataFrame(log_odds, 
                                            data[0][my_fold-1].drop(
                                                feat_remove, axis=1).columns, columns=['coef']).sort_values(by='coef', ascending=False)
                            else:
                                df_coef = pd.DataFrame(log_odds, 
                                            f_select, columns=['coef']).sort_values(by='coef', ascending=False)
                            # st.dataframe(df_coef)
                            fig5 = px.bar(df_coef, y=df_coef.index, x="coef",
                                          orientation='h')
                            st.plotly_chart(fig5)

                            
                    else:
                        st.empty()
                    
                    
                    st.header('Classification Report:')

                    st.text(reports[my_fold])
                    
                    if check_feat_select == 'disable':
                        X_test = data[2][my_fold-1].drop(feat_remove, axis=1)
                        y_test = data[2][my_fold-1][target]

                    else:
                        X_test = data[2][my_fold-1][f_select]
                        y_test = data[2][my_fold-1][target]

                        
                    pred_test = saved_models[my_fold-1].predict(X_test)
                    plot_confusion_matrix(saved_models[my_fold-1], X_test, y_test)

                    all_sample_title = 'Accuracy Score: {0}'.format(
                        saved_models[my_fold-1].score(X_test, y_test))
                    plt.title(all_sample_title, size=10)
                    st.header('Confusion matrix')
                    st.pyplot(plt)
                                    
                    if n_classes > 3:
                        st.header('ROC curve and ROC area for each class:')

                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()

                        y_val_test_bi = label_binarize(
                            y_test, classes=sorted(y_test.unique()))
                        y_val_pred_bi = label_binarize(
                            pred_test, classes=sorted(y_test.unique()))
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_val_test_bi[:, i], y_val_pred_bi[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])

                        # Plot of a ROC curve for a specific class
                        slide_class = st.selectbox(
                            'Label', y_test.unique())
                        
                        slide_class = int(float(slide_class))
                        if slide_class == -1 and missing=='yes':
                            slide_class = 0
                            
                        elif missing != 'yes':
                            slide_class -= 1
                        plt.figure(figsize=(5, 5))
                        plt.plot(
                            fpr[slide_class], tpr[slide_class], label='ROC curve (area = %0.2f)' % roc_auc[slide_class])
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        if slide_class == 0 and missing=='yes':
                            plt.title(
                                f'Receiver operating characteristic cluster {-1}')
                        elif missing == 'no':
                            plt.title(
                                f'Receiver operating characteristic cluster {slide_class+1}')
                        else:
                            plt.title(
                                f'Receiver operating characteristic cluster {slide_class}')

                        plt.legend(loc="lower right")
                        # plt.show()
                        st.pyplot(plt)
                        
                    elif n_classes <= 3:
                        st.header('ROC curve and ROC area for each class:')

                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()

                        y_val_test_bi = label_binarize(
                            y_test, classes=sorted(y_test.unique()))
                        y_val_pred_bi = label_binarize(
                            pred_test, classes=sorted(y_test.unique()))
                        for i in range(n_classes):
                            if n_classes == 2:
                                fpr[i], tpr[i], _ = roc_curve(
                                    y_test.astype('int'), pred_test.astype('int'))
                                roc_auc[i] = auc(fpr[i], tpr[i])

                            else:
                                fpr[i], tpr[i], _ = roc_curve(y_val_test_bi[:, i], y_val_pred_bi[:, i])
                                roc_auc[i] = auc(fpr[i], tpr[i])

                        # Plot of a ROC curve for a specific class
                        slide_class = st.selectbox(
                            'Label', y_test.unique())
                        if missing == 'yes':
                            slide_class = int(float(slide_class)) + 1
                        else:
                            slide_class = int(float(slide_class))
                        plt.figure(figsize=(5, 5))
                        plt.plot(
                            fpr[slide_class], tpr[slide_class], label='ROC curve (area = %0.2f)' % roc_auc[slide_class])
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        if missing == 'yes':
                            plt.title(
                                    f'Receiver operating characteristic cluster {slide_class-1}')
                        else:
                            plt.title(
                                f'Receiver operating characteristic cluster {slide_class}')
                        plt.legend(loc="lower right")
                        st.pyplot(plt)
                
