
from functools import partial


from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner


# def get_kdtree_filter(features):
#     return cKDTree(features)

def apply_ball_filter(query_points: np.ndarray, tree: cKDTree, max_tree_separation=5):
    # TODO experiment with epsilon for speed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.cKDTree.query_ball_point
    neighbor_indices = tree.query_ball_point(query_points, r=max_tree_separation, p=1)
    print(len(neighbor_indices), 'potential query neighbors found')
    return neighbor_indices



def get_batch_learner(batch_size, n_neighbors, X_train=None, y_train=None):
    # Specify our core estimator.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    preset_batch = partial(uncertainty_batch_sampling, n_instances=batch_size)

    # Specify our active learning model.
    learner = ActiveLearner(
        estimator=knn,
        X_training=X_train,
        y_training=y_train,
        query_strategy=preset_batch
    )
    return learner


def run_active_learning_iteration(batch_size, df, tree:cKDTree=None, max_tree_separation=15, max_tree_neighbors=10000):
    
    feature_cols = [col for col in df.columns.values if col.startswith('feat_pca')]

    # Isolate the non-training examples we'll be querying.
    X_train = df[df['has_label']][feature_cols].values
    y_train = df[df['has_label']]['label'].values

    if tree is None:
        X_pool = df[~df['has_label']][feature_cols].values
    else:
        print('using tree')
        X_pool_indices = tree.query_ball_point(X_train, r=max_tree_separation, p=1, return_sorted=True)
        # X_pool_indices is array of list of indices
        X_pool_indices = np.concatenate(X_pool_indices)
        # exclude any indices already labelled
        X_pool_indices = np.setdiff1d(X_pool_indices, df[df['has_label']].index)
        # keep only the first k
        X_pool_indices = X_pool_indices[:max_tree_neighbors]
        # print(X_pool_indices)
        X_pool = df.loc[X_pool_indices][feature_cols].values
    print('candidates to query: ', len(X_pool))

    print('fitting learner')
    learner = get_batch_learner(
        batch_size=batch_size, 
        X_train=X_train,
        y_train=y_train,
        n_neighbors=min(len(X_train), 5)
    )

    print('querying learner')
    query_indices, query_instances = learner.query(X_pool)
    return query_indices, learner

def dummy_label_query_indices(query_indices, df):
    df.loc[query_indices, 'has_label'] = True
    df.loc[query_indices, 'label'] = df.loc[query_indices].apply(get_dummy_label, axis=1)
    return df

def get_dummy_label(galaxy):
    return int(galaxy['feat_pca_0'] > 0)
