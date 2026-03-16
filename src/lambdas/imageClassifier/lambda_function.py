import json
import pickle
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import os
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = os.environ.get("ENDPOINT_NAME")
s3 = boto3.client("s3")
client = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):
    event_body = event['body']
    data = event_body['image_data']
    img = pickle.loads(data.encode('latin-1'))
    
    print("Image deserialized")

    # Transform the input to be fed into the network
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    img_transformed = transform(img)
    print("Image transformed")

    image = np.array(img_transformed).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    
    # Pickle the image to pass it on to the predictor
    image_p = pickle.dumps(image)
    print("Image pickled")

    print("Invoking the predictor...")
    # Make a prediction with the boto3 client:
    inference = client.invoke_endpoint(
        EndpointName=ENDPOINT, Body=image_p, ContentType="application/x-npy"
    )
    print(f"... response {inference} received")
    
    inferences = inference["Body"].read().decode().replace("[", "").replace("]", "")
    print(f"Raw inference {inferences} received")

    s = nn.Sigmoid()
    prediction = s(torch.Tensor([float(inferences)])).item()

    event_body["inferences"] = prediction
    print(f"Prediction {event_body['inferences']} received")

    return {"statusCode": 200, "body": json.dumps(event_body)}
