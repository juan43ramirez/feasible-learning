# Notes on metrics and meters

We use `Metric` and `Meter` objects to carry out measurements of various values of interest during training (and during model evaluation). Examples of these values are the loss, the violation of certain constraints, the overall or per-class accuracy, etc.

A `Metric` object executes the actual measurement of the value. For example, the
`forward` method of the `PerClassTop1Accuracy` metric calculates the top-1 accuracy for
each class based on pre-computed `predictions` and `targets`. In this case, the metric
actually calculates multiple values, one accuracy estimate for each class. The return
for the `forward` method is a dictionary. The possible keys that this dictionary may
contain are gathered in the `known_returns` attribute of the `Metric` class. Note that
the returned dictionary may contain only a subset of the keys in `known_returns`.

It is typically necessary to aggregate the values calculated by a `Metric` object over
multiple mini-batches for logging or other purposes. This is the role of a `Meter`
object. A `Meter` is used to compute statistics of a streaming series of values. For
example, the `AverageMeter` computes the (online) average of the values it receives.

It is required to provide a `Meter` type when instantiating a metric. For example, one
might want to track the _average_ accuracy per class across the entire dataset, or the
_maximum_ constraint violation across all samples.
