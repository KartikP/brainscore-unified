# Restore language compatibility with current upstream

Kartik authorized the upstream comparison and fixes. He owns the manual release;
publication, remote release CI and deployment are outside this work.

Clean language main `2c4744e` uses full-context forward passes for neural
recording. The v2 helper always enables the token cache. On the full 243-sentence
Pereira ridge case, the same FP32 checkpoint produces 160,909 different
activation values, with a maximum difference of 0.000244140625. Normalized scores
are 0.8353256254695343 upstream and 0.835324723971699 in v2.

Restore upstream's cache choice: use it only for behavioral tasks without neural
recording. Keeping the optimization and loosening legacy tolerances would break
the exact-compatibility requirement. Adding a new cache option would expand the
API without fixing the default. Native FP32 budgets stay unchanged.

Test neural outputs against independent full-context inference, including calls
that collect behavior and activations together. Check behavioral-only inference
still uses its cache. Rerun all four language cases against clean upstream, then
check adapter and native routes. Upstream names the old random-sentence linear
variant `linear-shuffle`; record that mapping. Ridge remains the current protocol.

Separate workers record source origins, checkpoint hashes, metric inputs and
scores. Vision master uses sklearn 1.5.2; language main's grouped folds require
the existing 1.7.2 profile. These are separate environment qualifications.
