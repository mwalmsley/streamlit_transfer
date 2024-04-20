
from scipy.spatial import cKDTree
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import shared
import learning

def test_ckdtree():
    df, features = shared.prepare_data()

    print('fitting')
    # clf = KNeighborsClassifier(n_neighbors=5).fit(X=features, y=np.random.randint(0, 2, size=len(features)))

    tree = cKDTree(features)
    print('tree constructed')
    # TODO experiment with epsilon for speed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html#scipy.spatial.cKDTree.query_ball_point
    print(tree.query_ball_point(features[500:510], r=5, p=1))



def test_ranked_active_learning():

    # Load our data.
    df, _ = shared.prepare_data()
    # df = df.sample(100000).reset_index(drop=True)

    df['has_label'] = False

    n_labeled_examples = 1  # a single initial query galaxy

    # pretend a random galaxy is labeled
    initial_training_index = np.random.randint(low=0, high=len(df), size=n_labeled_examples)
    df.loc[initial_training_index, 'has_label'] = True
    df.loc[initial_training_index, 'label'] = learning.get_dummy_label(df.loc[initial_training_index])

    n_queries = 10
    batch_size = 20

    feature_cols = [col for col in df.columns.values if col.startswith('feat_pca')]
    tree = cKDTree(df[feature_cols].values)

    performance_history = []
    for index in range(n_queries):

        print(df['has_label'].sum(), 'training examples')

        df, learner = learning.run_active_learning_iteration(batch_size, df, tree=tree)

        # Calculate and report our model's accuracy
        # debugging only

        print('scoring (debug only)')
        df_pred = df.sample(50000)
        model_accuracy = learner.score(df_pred[feature_cols].values, df_pred.apply(learning.get_dummy_label, axis=1))
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)


    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(performance_history)
    ax.scatter(range(len(performance_history)), performance_history, s=13)

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=n_queries + 3, integer=True))
    ax.xaxis.grid(True)

    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(bottom=0, top=1)
    ax.yaxis.grid(True, linestyle='--', alpha=1/2)

    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')

    plt.show()



if __name__ == '__main__':

    test_ranked_active_learning()