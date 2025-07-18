# TO_DEVS: BURN AFTER READING

Delete this section of the document once the app development is done, before publishing the repository.

---
This skeleton code is a scaffolding for Python-based CLAMS app development. Specifically, it contains

1. `app.py` and `metadata.py` to write the app
1. `requirements.txt` to specify python dependencies
1. `Containerfile` to containerize the app and specify system dependencies
1. `.gitignore` and `.dockerignore` files listing commonly ignored files
1. an empty `LICENSE` file to replace with an actual license information of the app
1. This `README.md` file for additional information not specified in the general user manual at https://apps.clams.ai/clamsapp
1. A number of GitHub Actions workflows for issue/bugreport management
1. A GHA workflow to publish app images upon any push of a git tag
   * **NOTE**: All GHA workflows included are designed to only work in repositories under `clamsproject` organization.

Before pushing your first commit, please make sure to delete this section of the document.

Then use the following section to document any additional information specific to this app. If your app works significantly different from what's described in the generic readme file, be as specific as possible.


> **warning**
> TO_DEVS: Delete these `TO_DEVS` notes and warnings before publishing the repository.

---

# Moondream Classifier

> **warning**
> TO_DEVS: Again, delete these `TO_DEVS` notes and warnings before publishing the repository.

## Description

> **note**
> TO_DEVS: A brief description of the app, expected behavior, underlying software/library/technology, etc.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

Below is a list of additional information specific to this app.

> **note**
> TO_DEVS: Below is a list of additional information specific to this app.


### System requirements

> **note**
> TO_DEVS: Any system-level software required to run this app. Usually include some of the following:
> * supported OS and CPU architectures
> * usage of GPU
> * system package names (e.g. `ffmpeg`, `libav`, `libopencv-dev`, etc.)
> * some example code snippet to install them on Debian/Ubuntu (because our base images are based on Debian)
>     * e.g. `apt-get update && apt-get install -y <package-name>`

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) or the [`metadata.py`](metadata.py) file in this repository.

> **warning**
> TO_DEVS: If you're not developing this app for publishing on the CLAMS App Directory, the above paragraph is not applicable. Feel free to delete or change it.

> **note**
> TO_DEVS: all runtime parameters are supported to be VERY METICULOUSLY documented in the app's `metadata.py` file. However for some reason, if you need to use this space to elaborate what's already documented in `metadata.py`, feel free to do so.
