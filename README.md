In this DESI Legacy Imaging Surveys Point Source Catalog (LS-PSC), we provide a machine-learning score for the probability of a LS target being an unresolved, point source. This is a morphological classifier based on the LS photometry, and provide classification for $3\times10^9$ LS targets. LS-PSC is integrated into the real-time alert stream of [the La Silla Schmidt Southern Survey (LS4)](https://sites.northwestern.edu/ls4/).

# Query sources with API
Scores can be queried from the [database](https://ls-xgboost.lbl.gov) via API requests.

```python
import requests

def get_sources(ra: float, dec: float, radius: float, mag_limit: float=None) -> dict:
    """
    Cross match to the LS-PSC catalog and get score sources.

    Parameters
    ----------
    ra, dec : float
        Coordinates in degrees.
    radius: float
        Search radius in degrees.
    mag_limit: float, optional
        Magnitude limit for the search (in white_mag)

    Returns
    -------
    dict
        A dictionary containing the sources found within the searching radius.
    """
    base_url = "https://ls-xgboost.lbl.gov/getsources"
    url = f"{base_url}/{ra}/{dec}/{radius}"
    if mag_limit is not None:
        url += f"/{mag_limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

```

# Citation
If you use LS-PSC in your research, please cite the following publication:
- Liu et al. (2025): [arXiv](https://arxiv.org/abs/2505.17174)
