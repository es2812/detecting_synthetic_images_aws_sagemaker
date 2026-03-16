
# Filter
import json

THRESHOLD = .60

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inference = float(json.loads(event['body'])['inferences'])
    print(inference)
    
    # We check what the predicted class is, if it's less than 0.5, then it's detecting the image is Fake, if it returns
    # over 0.5, it's detecting the image is Real
    confidence = 0.0
    if inference >= 0.5:
        confidence = inference
        print(f"The network predicted a REAL image with {round((confidence*100), 2)}% confidence")
    else:
        confidence = 1-inference
        print(f"The network predicted a FAKE image with {round((confidence*100), 2)}% confidence")

    # Check if the confidence is above THRESHOLD
    meets_threshold = confidence > THRESHOLD
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }