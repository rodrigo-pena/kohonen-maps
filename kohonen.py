# -*- coding: utf-8 -*-
"""
Module for implementing self-organizing maps (SOM), a.k.a. Kohonen maps.
"""

import numpy as np
import matplotlib.pylab as plb


def name2digits(name, hash_path='data/hash.mat'):
    r"""
    Convert string into pseudo-random selection of 4 digits from 0-9.

    Parameters
    ----------
    name : string
        String to be converted

    Returns
    -------
    digits : array_like
        Array of pseudo-randomly generated digits
    hash_path : string
        Path to the file hash.mat

    Notes
    -----
    Requires file hash.mat to be located under the data/ folder.
    Only the first 25 characters of the input string are considered.

    Examples
    --------
    >>> from kohonen import name2digits
    >>> d = name2digits('hello')
    >>> print(d)
    [4, 7, 8, 9]

    """

    name = name.lower()
    n = len(name)
    if n > 25:
        name = name[0:25]

    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                     53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    # Create seed based on the Unicode code point of each character in name.
    seed = 0.0
    for i in range(n):
        seed += prime_numbers[i] * ord(name[i]) * 2.0**(i + 1)

    # Import table containing 4-groupings of the digits 0-9
    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('data/hash.mat', struct_as_record=True)
    digits_choices = Data['x']
    n_choices = digits_choices.shape[0]

    # Select a pseudo-random row of this table
    row = np.mod(seed, n_choices).astype('int')

    digits = np.sort(digits_choices[row, :])
    return digits


def preprocess_mnist(data_path='data/data.txt', label_path='data/labels.txt',
                     name='Rodrigo Pena'):
    r"""
    Selecting a pseudo-random subset of digits in the MNIST dataset.

    Parameters
    ----------
    data_path : string
        Path to the data file
    label_path : string
        Path to the label file
    name : string
        Seed for the pseudo-random selector.

    Returns
    -------
    data : array_like
        Subset corresponding to 4 pseudo-randomly selected digits.
    digits : array_like
        List of digits selected by the pseudo-random process.

    Example
    -------
    >>> from kohonen import preprocess_mnist
    >>> full_data = np.array(np.loadtxt('data/data.txt'))
    >>> full_data.shape
    [5000, 784]
    >>> data, selected_digits = preprocess_mnist()
    >>> data.shape
    [2000, 784]
    >>> print(selected_digits)
    [5, 6, 7, 9]

    """

    # Load in data and labels
    data = np.array(np.loadtxt(data_path))
    labels = np.loadtxt(label_path)

    # Select the 4 digits that should be used
    digits = name2digits(name)

    # Restrict full dataset to subset of selected digits
    data = data[np.logical_or.reduce([labels == x for x in digits]), :]

    return data, digits


def gauss(x, p):
    r"""
    (Normalized) Gaussian kernel N(x).

    Parameters
    ----------
    x : array_like
        Function evaluation point.
    p : 2D list or ndarray
        p[0] is the mean, p[1] is the standard deviation of N(x)

    Returns
    -------
    N : array_like
        Evaluation of the Gauss function at x.

    Notes
    -----
    N(x) is normalized such that N(p[0]) = 1.

    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))


def som_step(centers, datapoint, neighbor_matrix, eta, sigma):
    r"""
    Single step of the sequential learning for a self-organizing map (SOM).

    Parameters
    ----------
    centers : array_like
        Cluster centers. Format: "cluster-number"-by-"coordinates"
    datapoint : array_like
        Datapoint presented in the current timestep.
    neighbor : array_like
        Array encoding the coordinates of each cluster center.
    eta : float
        Learning rate
    sigma : float
        Width of the gaussian neighborhood function.

    Returns
    -------
    None

    Notes
    -----
    The input variable centers is modified inside the function.

    """

    # Initialization
    n_centers, dim = centers.shape

    # Find the best matching unit (bmu)
    bmu = np.argmin(np.sum((centers - datapoint) ** 2, axis=1))

    # Find coordinates of the bmu on the map
    a_bmu, b_bmu = np.nonzero(neighbor_matrix == bmu)

    # Update all units
    for j in range(n_centers):
        # Find coordinates of the current unit
        a, b = np.nonzero(neighbor_matrix == j)

        # Calculate the discounting factor of the neighborhood function
        dist = np.sqrt((a_bmu - a)**2 + (b_bmu - b)**2)
        disc = gauss(dist, [0, sigma])

        # update weights
        centers[j, :] += disc * eta * (datapoint - centers[j, :])


def kohonen(data, size_k=6, sigma=2.0, eta=0.9, maxit=5000, tol=0):
    r"""
    Compute a self-organizing map (SOM) fitting the input data.

    Parameters
    ----------
    size_k : int
        One of the dimensions of the square grid representing the map.
    sigma : float
        Standard deviation of the Gaussian kernel used as neighborhood
        function.
    eta : float
        Learning rate
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance on the successive (normalized) energy difference between
        iterations.

    Returns
    -------
    centers: array_like
        Array of coordinates of the fitted cluster centers.

    Example
    -------
    >>> from kohonen import preprocess_mnist, kohonen
    >>> data, _ = preprocess_mnist()
    >>> centers = kohonen(data)

    """

    n_samples, dim = data.shape
    data_range = np.max(data)
    n_centers = int(size_k ** 2)
    centers = np.random.rand(n_centers, dim) * data_range
    neighbor_matrix = np.arange(n_centers).reshape((size_k, size_k))

    # Set the random order in which the datapoints should be presented
    order = np.arange(maxit) % n_samples
    np.random.shuffle(order)

    # Run SOM steps
    for i in order:
        som_step(centers, data[i, :], neighbor_matrix, eta, sigma)

    return centers


def visualize_map(centers):
    r"""
    Visualize the SOM output by function kohonen().

    Parameters
    ----------
    centers : array_like
        Cluster centers. Format: "cluster-number"-by-"coordinates"

    Returns
    -------
    None

    Notes
    -----
    The SOM is plotted as a grid of images, with one image per cluster center.
    Works well if the map was fit to an image datasets, such as MNIST.

    """

    size_k = int(np.sqrt(centers.shape[0]))
    dim = int(np.sqrt(centers.shape[1]))

    plb.close('all')

    for i in range(centers.shape[0]):
        plb.subplot(size_k, size_k, i + 1)

        plb.imshow(np.reshape(centers[i, :], [dim, dim]),
                   interpolation='bilinear')
        plb.axis('off')

    plb.show()
    plb.draw()


if __name__ == "__main__":
    r"""
    Run kohonen() with default values.

    """

    data_path = 'data/data.txt'
    label_path = 'data/labels.txt'
    name = 'Rodrigo Pena'
    size_k = 6
    sigma = 2.0
    eta = 0.9
    maxit = 5000

    data, _ = preprocess_mnist(data_path=data_path, label_path=label_path,
                               name=name)

    centers = kohonen(data, size_k=size_k, sigma=sigma, eta=eta, maxit=maxit)

    visualize_map(centers)
