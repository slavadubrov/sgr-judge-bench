# Judge benchmark results

Published-rate usage estimates are not invoice-confirmed charges. Failed outputs remain incorrect in primary quality metrics. Unattempted cases are reported separately; incomplete runs cannot establish a quality comparison.

| phase | dataset | judge | profile | attempted | macro_f1 | coverage | brier | p95_s | usd_per_1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | ragtruth | deepseek-json | D | 900 | 0.7303 | 1.0000 | unknown | 1.1308 | 0.1108 |
| test | ragtruth | glm-current | D | 900 | 0.7661 | 0.9944 | unknown | 2.5608 | 0.0970 |
| test | ragtruth | jev | P | 900 | 0.6667 | 1.0000 | 0.2039 | 0.4914 | 0.0369 |
| test | ragtruth | luna | D | 900 | 0.6539 | 1.0000 | unknown | 1.4171 | 0.1566 |

See [the results](README.md) for methods, paired confidence intervals, failure examples and limits.
