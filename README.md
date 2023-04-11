# musepsf

Small package to measure the PSF of a MUSE datacube usiong a cross-convolution technique to
compare an image extracted from the cube to a reference image with known PSF. The method is
an adapted and simplified version of the procedure provided in [Bacon et al. 2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...1B/abstract)


## Usage

Inatallation:

Clone this repository using `git clone https://github.com/cloud182/musepsf.git` and installi it
locally via `python -m pip install -e .`.

To see an exaple of how to use the package to measure the PSF of a MUSE image, please check the
`examples` directory.

## License

`musepsf` was created by Enrico Congiu. It is licensed under the terms of the MIT license.

## Credits

`musepsf` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
