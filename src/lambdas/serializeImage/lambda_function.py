# Serializer
import json
import boto3
from PIL import Image
import numpy as np
import pickle

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.jpg
    with open('/tmp/image.jpg', 'wb') as data:
        s3.download_fileobj(bucket, key, data)
    
    # We read the image with PIL
    img = Image.open('/tmp/image.jpg')
    # We convert the image to a numpy array
    img_np = np.array(img)

    # We serialize the array with picke
    serialized = pickle.dumps(img_np).decode('latin-1')

    # Pass the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": serialized,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }