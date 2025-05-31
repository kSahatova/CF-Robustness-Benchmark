import numpy as np
import matplotlib.pyplot as plt


def generate_fake_samples(x, generator):
    """Use the input generator to generate samples."""
    return generator.predict(x, verbose=0)


def data_stream(x, y=None, batch_size=500):
    """Generate batches until exhaustion of the input data."""
    n_train = x.shape[0]
    if y is not None:
        assert n_train == len(y)
    n_complete_batches, leftover = divmod(n_train, batch_size)
    n_batches = n_complete_batches + bool(leftover)

    perm = np.random.permutation(n_train)
    for i in range(n_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        if y is not None:
            output = (x[batch_idx], y[batch_idx])
        else:
            output = x[batch_idx]
        yield output


def infinite_data_stream(x, y=None, batch_size=500):
    """Infinite batch generator."""
    batches = data_stream(x, y, batch_size=batch_size)
    while True:
        try:
            yield next(batches)
        except StopIteration:
            batches = data_stream(x, y, batch_size=batch_size)
            yield next(batches)


def plot_generated_images(x, n_plots=5, n_plots_per_row=5, residuals=False, save_path=''):
    """Plot several images in a grid."""
    x = x.reshape((x.shape[0], 28, 28))
    lower_bound = -1 if residuals else 0
    upper_bound = 1
    for i in range(n_plots):
        ax = plt.subplot(5, n_plots_per_row, 1 + i)
        ax.axis('off')
        ax.imshow(x[i], vmin=lower_bound, vmax=upper_bound, cmap="gray")
    plt.savefig(save_path)
    plt.show()


def compute_reconstruction_error(x, autoencoder):
    """Compute the reconstruction error for a given autoencoder and data points."""
    preds = autoencoder.predict(x, verbose=0)
    preds_flat = preds.reshape((preds.shape[0], -1))
    x_flat = x.reshape((x.shape[0], -1))
    return np.linalg.norm(x_flat - preds_flat, axis=1)


def format_metric(metric):
    """Return a formatted version of a metric, with the confidence interval."""
    return f"{metric.mean():.3f} ± {1.96*metric.std()/np.sqrt(len(metric)):.3f}"


def compute_metrics(samples, counterfactuals, latencies, classifier, autoencoder,
                    desired_class, batch_latency=None):
    """ Summarize the relevant metrics in a dictionary. """
    reconstruction_error = compute_reconstruction_error(counterfactuals, autoencoder)

    delta = np.abs(samples-counterfactuals)
    l1_distances = delta.reshape(delta.shape[0], -1).sum(axis=1)

    prediction_gain = (
        classifier.predict(counterfactuals, verbose=0)[:, desired_class] -
        classifier.predict(samples, verbose=0)[:, desired_class]
    )

    metrics = dict()
    metrics["reconstruction_error"] = format_metric(reconstruction_error)
    metrics["prediction_gain"] = format_metric(prediction_gain)
    metrics["sparsity"] = format_metric(l1_distances)
    metrics["latency"] = format_metric(latencies)
    batch_latency = batch_latency if batch_latency else sum(latencies)
    metrics["latency_batch"] = f"{batch_latency:.3f}"

    return metrics
