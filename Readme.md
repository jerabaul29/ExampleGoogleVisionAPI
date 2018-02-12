# An example of use of Google Cloud Face API

This repository contains code to use the google cloud face API. There is an associated post, available here:

https://folk.uio.no/jeanra/Informatics/GoogleCloudExampleFaceAPI.html

You will need to set up a Google Cloud account, create a project, enable the API, and export credentials as indicated in the post to be able to use the code.

## Setup

This is developed for Ubuntu 16.04 with Python 2.7. My computer has a webcam available on */dev/video0*. You can adapt the code to use another video input.

To get the code ready to work:

- Copy the code on your computer, for example by cloning this repo.
- Make sure to have the necessary packages: ffmpeg, v4l-utils.
- Make sure you have the necessary python modules: scipy, matplotlib, google, a (correctly) installed PIL.
- Adapt the *detect_faces.sh* script with the right path to your JSON credential.
- Make the script executable: chmod +x detect_faces.sh

## Execution

Simply:

```
./detect_faces.sh
```
