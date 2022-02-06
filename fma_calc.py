import pandas as pd
import numpy as np

def extract_measure_count(feature_target, measure_target):
    '''
    Extract count channels
    return number
    '''
    #count feature
    feature_count = 0
    for f in features.columns:
        if f[0] == feature_target and f[1] == measure_target:
            feature_count += 1
        else:
            if feature_count > 0:
                return feature_count

    return feature_count

def similarity(A, B, method = 'coseno'):
    '''
    coseno similarity
    https://miro.medium.com/max/644/0*7dWSG0979AvY4Wo2
    '''
    if method == 'coseno':
        return np.sum(A * B) / (np.sqrt(np.sum(A**2)) * np.sqrt(np.sum(B**2)))
    elif method == 'euclidean':
        return 0.5 * (variance(A - B) / (variance(A) + variance(B)))
    return 0

def variance(A):
    return (np.sum((A - np.mean(A))**2)/len(A))

def correlation(col1_data, col2_data):
    '''
    calc correlation between two features
    
    use Pearson’s correlation coefficient formula
    https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
    '''
    n = len(col1_data)
    col1_sum = np.sum(col1_data)
    col2_sum = np.sum(col2_data)
    
    return (n*np.sum(col1_data*col2_data) - col1_sum * col2_sum) / np.sqrt((n * np.sum(col1_data**2) - col1_sum**2) * (n * np.sum(col2_data**2) - col2_sum**2))

def intra_list_similarity(target, predict):
    return (np.sum(target) * (np.sum(predict)**similarity(target, predict)))/2

def measure_similarity(data_features, track_id_target, features_label = None, measures_type = None, method = 'coseno'):
    '''
    measure similarity for all features together
    '''
    
    if features_label != None and measures_type != None:
        X = data_features.loc[:, (features_label, measures_type)]
    elif features_label != None:
        X = data_features.loc[:, features_label]
    else:
        X = data_features
    
    col = pd.Series(dtype='float64')
    for track_id in X.index:
        col = col.append(pd.Series([similarity(X.loc[track_id_target], X.loc[track_id], method=method)], index=[track_id], dtype='float64'))
    return col

def measure_similarity_feature(data_features, track_id_target, features_by = ['chroma_cens', 'chroma_cqt', 'chroma_stft', 'mfcc', 'rmse', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'spectral_rolloff', 'tonnetz', 'zcr'], measures_type = ['kurtosis', 'max', 'mean', 'median', 'min', 'skew', 'std'], method = 'coseno'):
    '''
    measure similarity per feature
    '''
    data = pd.DataFrame()
    for feature_by in features_by:
        if measures_type == None:
            X = data_features.loc[:, feature_by]
        else:
            X = data_features.loc[:, (feature_by, measures_type)]

        col = pd.Series(dtype='float64')
        for track_id in X.index:
            col = col.append(pd.Series([similarity(X.loc[track_id_target], X.loc[track_id], method=method)], index=[track_id], dtype='float64'))
        data[feature_by] = col
    
    #sum all features similarities
    data['similarity_sum'] = data.sum(axis=1)
    
    #calc simple mean for all features similarities
    data['similarity_mean'] = data['similarity_sum']/len(features_by)
    
    return data

def measure_correlation(data_features):
    '''calc correlation between all features'''
    data = pd.DataFrame()
    for col1 in data_features.columns:
        col = pd.Series(dtype='float64')
        for col2 in data_features.columns:
            col = col.append(pd.Series([correlation(data_features[col1], data_features[col2])], index=[col2], dtype='float64'))
        data[col1] = col
    return data

def drop_correlation(data_features, corr = 0.9):
    '''drop columns by correlation'''
    
    columns_drop = []
    all_ready_calc = []
    for col1 in data_features.columns:
        all_ready_calc.append(col1)
        
        for col2 in data_features.columns:
            if col1 != col2:
                calc = correlation(data_features[col1], data_features[col2])
            
                if not col2 in all_ready_calc and not col2 in columns_drop and calc >= corr:
                    columns_drop.append(col2)
        
    return columns_drop