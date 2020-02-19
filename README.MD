# Example Deployment of an ML Model via Docker

This repo contains code to train a machine learning model to predict Income using the [Adult Census dataset](https://archive.ics.uci.edu/ml/datasets/Adult), then to build and dockerise a REST API endpoint that allows you to perform real-time inference against the trained model. This is a popular means of deploying ML models; you might have heard of frameworks for deployment of machine learning models like [MLFlow](https://mlflow.org/) or the [AzureML SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) - fundamentally, this is exactly what they're doing (though the docker images they build are much more complicated, including logging and more robust error handling and stuff).

### What this repo is not

Any kind of production ready or 'optimal' solution. It's a quick demo of how things work; it should highlight how to implement this yourself if you want to (and it's often a good idea, because the frameworks do some weird stuff like bloat the image pretty badly or deploy multiple containers hosting stuff you don't remotely need), but some serious additional work is necessary before this would be safe to use in a production system.

### How's the deployment work?

I assume you know how the models themselves are trained; once you have the model saved to disk (as a pickled scikit-learn object or a keras h5 file or whatever it is) the first step is to build a [Flask](https://palletsprojects.com/p/flask/) application that loads up the model, then exposes a route to the `.predict()` function (or whatever your particular model-flavour's equivalent is). Data that's POSTed to this route gets passed to `.predict()`, and the output jsonified and returned.

Step 2, use [Gunicorn](https://gunicorn.org/) to serve the Flask app - because Flask's own bundled server isn't really good enough. Gunicorn isn't the best thing for doing the initial handling of client connections though, so the usual method is to only expose Gunicorn on localhost and then pass it traffic with...

Step 3, Apache/Nginx. Either works fine, although I'm using Apache because I'm familiar with it. All this layer does is listen on 0.0.0.0 and pass all the traffic to Gunicorn using ProxyPass.

### Where does Docker come into it?

Pretty much all the usual benefits of a container. Packaging all of the above into a docker image is a convenient way of deploying stuff; it lets you host a Container instance rather than a full server, it's easy to spin up a bunch of them behind a load balancer, they're easy to manage and so on.

Containerising the deployment is reasonably straightforward. You start from a base image; I've chosen Ubuntu so that the commands in the Dockerfile are familiar to people, but a more common choice would be Alpine Linux. All it does is run the usual upgrades, then copy over the .pkl file that's produced by train.py and the various components of the deployment (app.py to serve the Flask app, model.conf to configure apache, requirements.txt to define all the python dependencies, servemodel.sh to boot everything up). The final CMD command runs the .sh script that boots up gunicorn and apache, and hey presto.

### How do I make it all do the thing?

1. Clone the repo:

`git clone https://github.com/djrscallu/ml-deploy-docker`

2. Train the model:

`python3 ./train.py`

3. Build the docker image

`sudo docker build -t djrscally/ml .`

4. Run the container

`sudo docker run -d -p 5000:5000 djrscally/ml`

And that's it; the model will be available at http://localhost:5000. You can try it out by passing some input to the model with the [requests] package, like so:

```Python
import requests

headers = {
  'Content-Type':'application/json'
}

inputs = """{
  "columns":[
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country"
  ],
  "data":[
    [
      25,
      " Private",
      32275,
      " Some-college",
      10,
      " Married-civ-spouse",
      " Exec-managerial",
      " Wife",
      " Other",
      " Female",
      0,
      0,
      40,
      " United-States"
    ]
  ]
}"""

requests.post('http://localhost:5000/predict', headers=headers, data=inputs).text
```

That's forwarded from your host to port 5000 on the container, which is where Apache is listening. Apache passes it to gunicorn at port 8000, and out pops the model's prediction: `{"predictions":[" <=50K"]}`
