# time-series-transformers

scikit-learn transformers for working with time series data.

`time-series-transformers` is a collection of [scikit-learn](https://scikit-learn.org)-compatible
transformers for common time series preprocessing tasks — scaling, log and signed-power
transforms, differencing, detrending, and trend extraction. Every transformer operates
directly on `pandas` DataFrames and preserves their index and column structure, follows the
familiar `fit` / `transform` API, and — where it makes sense — implements `inverse_transform`
so you can map results back to the original scale.

## Installation

```bash
pip install time-series-transformers
```

## Usage

All transformers accept and return `pandas` DataFrames:

```python
import pandas as pd

from time_series_transformers import (
    DifferenceTransformer,
    InvertiblePipeline,
    LogTransformer,
)

data = pd.DataFrame({"price": [100.0, 102.0, 101.5, 105.0, 110.0]})

# Chain transforms; InvertiblePipeline undoes them in reverse order.
pipeline = InvertiblePipeline(
    [
        ("log", LogTransformer()),
        ("difference", DifferenceTransformer()),
    ]
)

transformed = pipeline.fit_transform(data)
recovered = pipeline.inverse_transform(transformed)  # back to the original prices
```

The package provides:

- **Scaling** — `PandasStandardScaler`, `PandasMinMaxScaler`
- **Transforms** — `LogTransformer` (log / signed-power)
- **Stationarity** — `DifferenceTransformer`, `DetrendTransformer`, `HamiltonFilterTransformer`
- **Composition** — `InvertiblePipeline`, `DataFrameFeatureUnion`

## Contributing

Contributions are welcome! This project uses [pixi](https://pixi.sh) to manage its
development environment.

1. Fork and clone the repository.
2. Install the environment:
   ```bash
   pixi install
   ```
3. Create a feature branch and make your changes, adding tests where appropriate.
4. Run the checks before opening a pull request:
   ```bash
   pixi run test         # run the test suite
   pixi run mypy         # type-check
   pixi run pre-commit   # lint and format
   ```
5. Open a pull request against `main` with a clear description of your change.

Please make sure the test suite passes and new code is covered by tests.

## License

Released under the [MIT License](LICENSE).
