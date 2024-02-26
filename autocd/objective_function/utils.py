import numpy as np

def get_markov_boundary(dag):
    """Compute the Markov Boundary for each node in the graph.

    Parameters
    ----------
    dag: numpy array shape = (n_features, n_features)
        estimated causal adjacency matrix

    Returns
    -------
    List containing the Markov Boundary for each node.
    """
    n_features = dag.shape[0]
    markov_boundaries = [None] * n_features

    for node in range(n_features):
        to_node = np.where(dag[:, node] == 1)[0]
        from_node = np.where(dag[node, :] == 1)[0]
        undirected = np.intersect1d(
            to_node, from_node
        )  # Filter out same parents of the children and of the target
        parents = np.setdiff1d(to_node, undirected)
        children = np.setdiff1d(from_node, undirected)

        spouses = []
        for child, _ in enumerate(children):
            to_child = np.where(dag[:, children[child]] == 1)[0]
            from_child = np.where(dag[children[child], :] == 1)[0]
            undirected_child = np.intersect1d(to_child, from_child)
            parents_child = np.setdiff1d(to_child, undirected_child)

            if parents_child.size == 0:
                continue
            else:
                spouses.extend(parents_child)

        spouses = np.setdiff1d(spouses, node).astype("int")
        markov_boundaries[node] = np.unique(
            np.concatenate((undirected, parents, children, spouses), axis=0)
        )

    return markov_boundaries


def empirical_prob(y):
    """Compute empirically the probability distribution.

    Parameters
    ----------
    y: numpy.ndarray shape = (n_samples, 1) containing ints
        target variable

    Returns
    -------
    Proportion of unique variables and index of the maximum value.
    """
    A = np.bincount(y)
    probs = A / len(y)
    max_prob = np.argmax(probs)

    return probs, max_prob


def mutual_information(is_cat, node, y, yhat):
    """Metric that measures goodness of fit between the estimated variable and the true variable.

    Parameters
    ----------
    is_cat: numpy.ndarray shape = (n_features, 1) containing booleans
        True when the feature is categorical otherwise False
    node: int
        elected feature to measure the goodness of fit
    y: numpy.ndarray shape = (n_samples, 1)
        true variable
    yhat: numpy.ndarray shape = (n_samples, 1)
        estimated variable

    Returns
    -------
    Float which is the mutual information between the estimated and true variable.
    """
    y = np.array(y)
    yhat = np.array(yhat)
    if is_cat[node]:
        y = y.astype("int")
        yhat = yhat.astype("int")
        n_samples = len(y)

        A_y = np.bincount(y)
        p_Y = A_y / n_samples

        A_yhat = np.bincount(yhat)
        p_Yhat = A_yhat / n_samples

        subs = np.vstack((y, yhat)).T + 1
        A_joint = np.histogramdd(subs, np.max(subs, axis=0))[0]
        p_joint = A_joint / n_samples

        H_y = -np.nansum(p_Y * np.log(p_Y, where=p_Y > 0))
        H_yhat = -np.nansum(p_Yhat * np.log(p_Yhat, where=p_Yhat > 0))
        H_joint = -np.nansum(p_joint * np.log(p_joint, where=p_joint > 0))

        mutual_info = H_y + H_yhat - H_joint
    else:
        if np.std(y) == 0 or np.std(yhat) == 0:
            raise ValueError("Error with computing the standard deviation.")

        s_y = np.std(y, ddof=1)
        H_y = (1 / 2) * np.log(2 * np.pi * np.exp(1) * s_y**2)

        s_yhat = np.std(yhat, ddof=1)
        H_yhat = (1 / 2) * np.log(2 * np.pi * np.exp(1) * s_yhat**2)

        if np.array_equal(y, yhat):
            mutual_info = H_y
        else:
            n_Vars = 2
            K = np.cov(y, yhat, rowvar=False)
            c = (2 * np.pi * np.exp(1)) ** n_Vars
            H_joint = (1 / 2) * np.log(c * np.linalg.det(K))

            mutual_info = H_y + H_yhat - H_joint

    return mutual_info
