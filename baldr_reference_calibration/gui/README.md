# Baldr reference calibration GUI

This is a separate companion application. It does not modify the standalone
`baldr-reference-calibration` package and uses that installed package as its
physics and fitting engine.

## Install and run

Activate the environment in which `baldr-reference-calibration` is installed,
then install Streamlit and launch the app:

```bash
python -m pip install streamlit
python -m streamlit run app.py
```

Using `python -m streamlit` ensures that Streamlit and the calibration package
come from the same Python environment. The GUI also automatically discovers a
calibration repository when placed inside it or directly beside it.

The browser application has three tabs:

1. Edit the complete JSON configuration, regenerate clear and ZWFS theoretical
   intensities, and download them as FITS.
2. Either upload a measured clear/ZWFS FITS pair, or vary the true
   knife-edge/cold-stop alignment, detector crop, pupil offset, flux,
   background, and noise to generate one synthetically.
3. run automatic registration, cropping, coarse grid search, and fine alignment
   optimization; inspect the parameter recovery, crop, residual, and fit maps.

The theoretical detector configuration should normally retain `"crop": null`;
the GUI produces the smaller measurement subframe independently.
