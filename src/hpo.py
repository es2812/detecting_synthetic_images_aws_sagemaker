import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import os
from loguru import logger
import argparse

def test(model, test_loader, criterion, labels_ix, device):
    """Performs testing of the model.

    Args:
        model (Model): Model to test
        train_loader (DataLoader): Iterable which yields the data in tuples X, y in batches
        criterion (BCEWithLogitsLoss): Loss function, used to calculate the loss of the model
        labels_ix (dict): Dictionary that contains the class label as key, and the corresponding neuron index as item
    """
    model.eval()
    running_loss = 0
    running_corrects = 0
    running_true_positives = 0
    running_false_negatives = 0

    s = nn.Sigmoid()
    
    total_images = len(test_loader.dataset)
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        
        outputs = model(inputs)
        outputs = outputs.to(device)
        # We need to pass the labels as a (N, 1) tensor of type float
        labels_t = labels.reshape(len(labels), 1).type(torch.FloatTensor)
        labels_t=labels_t.to(device)
        
        # we calculate the loss and un-average it
        loss = criterion(outputs, labels_t)
        
        # the raw output needs to be converted into a probability with the activation function (Sigmoid)
        prob = s(outputs)
        # we then assign a class, if the probability is equal or over 50% then it's assigned class 1,
        # otherwise it's assigned class 0
        preds = torch.tensor([1 if x >= 0.5 else 0 for x in prob])
        preds = preds.to(device)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # true positives = images correctly predicted as REAL
        running_true_positives += torch.sum(preds[preds == labels.data] == labels_ix["REAL"])
        # false negatives = real images, uncorrectly predicted as FAKE
        running_false_negatives += torch.sum(preds[preds != labels.data] == labels_ix["FAKE"])
        
    total_loss = running_loss / total_images
    total_acc = running_corrects.double() / total_images
    total_recall = running_true_positives / (
        running_true_positives + running_false_negatives
    )

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    logger.info(f"Testing Recall: {total_recall}")


def train(model, train_loader, criterion, optimizer, labels_ix, device):
    """Performs training of the model for at most 100 epochs. It will stop early if it finds that
    the loss has increased between trainings.

    Args:
        model (Model): Model to train
        train_loader (DataLoader): Iterable which yields the data in tuples X, y in batches
        criterion (BCEWithLogitsLoss): Loss function, used to calculate the loss of the model
        optimizer (AdamOptimizer): Optimizer which serves to perform the weight updates to the model
        labels_ix (dict): Dictionary that contains the class label as key, and the corresponding neuron index as item

    Returns:
        Model: trained model
    """
    epochs = 100
    best_loss = 1e6
    
    s = nn.Sigmoid()

    for epoch in range(epochs):
        # for each epoch...
        logger.info(f"Training epoch: {epoch}/{epochs}")

        # we set the model in train mode
        model.train()

        # set to 0 the different metrics we will log to screen
        running_loss = 0.0
        running_corrects = 0
        running_true_positives = 0
        running_false_negatives = 0

        idx_batch = 0
        num_images = 0
        total_number_of_batches = len(train_loader)
        total_images = len(train_loader.dataset)
        for inputs, labels in train_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            # for each batch...

            # forward-pass
            outputs = model(inputs)
            outputs = outputs.to(device)
            # We need to pass the labels as a (N, 1) tensor of type float
            labels_t = labels.reshape(len(labels), 1).type(torch.FloatTensor)
            labels_t=labels_t.to(device)
            
            # we calculate the loss
            loss = criterion(outputs, labels_t)

            # reset gradient
            optimizer.zero_grad()

            # calculate the weight update and perform backwards-pass
            loss.backward()
            optimizer.step()

            # calculate the loss, un-average it, and sum it to the previous batches in the same epoch
            running_loss += loss.item() * inputs.size(0)

            # same with number of correct predictions
            # the raw output needs to be converted into a probability with the activation function (Sigmoid)
            prob = s(outputs)
            # we then assign a class, if the probability is equal or over 50% then it's assigned class 1,
            # otherwise it's assigned class 0
            preds = torch.tensor([1 if x >= 0.5 else 0 for x in prob])
            preds = preds.to(device)
            
            current_corrects = torch.sum(preds == labels.data)
            running_corrects += current_corrects

            # true positives = images correctly predicted as REAL
            current_true_positives = torch.sum(preds[preds == labels.data] == labels_ix["REAL"])
            running_true_positives += current_true_positives
            # false negatives = real images, uncorrectly predicted as FAKE
            current_false_negatives = torch.sum(preds[preds != labels.data] == labels_ix["FAKE"])
            running_false_negatives += current_false_negatives
            
            num_images += len(inputs)
            if idx_batch % 100 == 0:
                # every 100 batches we print a small status report
                logger.info(f"""[Batch {idx_batch}/{total_number_of_batches}] Processed {num_images}/{total_images} ({(num_images/total_images)*100:.2f}%)
                            Average Loss achieved in this batch: {loss.item():.2f}
                            Current Total Loss: {running_loss:.2f}
                            
                            Correctly classified images in this batch: {current_corrects}/{len(inputs)}
                            True positives in this batch: {current_true_positives}/{len(inputs)}
                            False negatives in this batch: {current_false_negatives}/{len(inputs)}
                            
                            Current accuracy: {(running_corrects/num_images)*100:.2f}%
                            Current recall: {(running_true_positives/(running_true_positives + running_false_negatives))*100:.2f}%
                    """)
            idx_batch += 1
                        
        epoch_loss = running_loss / total_images
        epoch_acc = running_corrects / total_images
        epoch_recall = running_true_positives / (
            running_true_positives + running_false_negatives
        )

        logger.info(
            f"""Epoch {epoch}/{epochs} finished.
                    Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Recall: {epoch_recall:.4f}"""
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
        else:
            logger.info(
                f"""Stopping training after {epoch} epochs, have found loss increasing.
                        Best loss achieved {best_loss}, Loss achieved in the last epoch: {epoch_loss}"""
            )
            break

    return model


def create_data_loaders(data_dir: str, batch_size: int) -> tuple[DataLoader, dict]:
    """Creates DataLoaders for usage of images as inputs in a Pytorch model.

    Takes from argument the path that data is placed at, this must contain a subfolders for each class,
    under which the images which belong to that given class are placed in .jpg format.

    Creates Datasets for train and test data from the S3 URI, and applies the
    transformations necessary.


    Args:
        data_dir (str): path where the train and test folders are
        batch_size (int): the number of samples to load per batch

    Returns:
        tuple[DataLoader, dict]: DataLoader, map of index to class labels
    """
    transformer = transforms.Compose(
        [
            #transforms.Resize((299, 299)), #inception_v3 image size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    data = ImageFolder(root=data_dir, transform=transformer)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader, data.class_to_idx


def net() -> models.Inception3:
    """Returns the model to train and test.

    Returns:
        models.Inception3: A modified version of the VGG19 pre-trained model
            with an output of 1 neuron.
    """
    model = models.vgg19_bn(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # we obtain the number of output features in VGG19
    num_features = model.classifier[-1].out_features
    # we add an output layer for our binary classification with 1 output neuron
    # (the correct output for BCEWithLogitsLoss)
    model.classifier = model.classifier.append(nn.Linear(num_features, 1))
    
    return model


def main(args):
    """Creates and trains the model, after which it tests the model against the test data.
    Finally it stores the model in the path given by arguments.

    Args:
        args (Namespace): Arguments given to the main function
    """
    logger.info(f"Creating a model with the following parameters:")
    logger.info("  Loss function: Binary Cross Entropy Loss")
    logger.info(f"  Optimizer: ADAM with learning rate: {args.learning_rate}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"Train data directory: {args.data_train}")
    logger.info(f"Test data directory: {args.data_test}")
    logger.info(f"Outputs will be written to: {args.output_dir}")
    logger.info(f"Model will be saved to: {args.model_dir}")

    train_loader, class_index_map = create_data_loaders(
        args.data_train, args.batch_size
    )
    test_loader, _ = create_data_loaders(args.data_test, args.batch_size)
    model = net()
    
    # I use BCEWithLogitsLoss since it includes the Sigmoid activation function + BCELoss
    criterion = nn.BCEWithLogitsLoss()
    # I will optimize only the last layer, which was the one added
    optimizer = optim.Adam(model.classifier[-1].parameters(), lr=args.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device {device}")
    
    logger.info("Starting Model Training")
    model=model.to(device)
    model = train(model, train_loader, criterion, optimizer, class_index_map, device)

    logger.info("Testing model")
    test(model, test_loader, criterion, class_index_map, device)

    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--data_train", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--data_test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    args = parser.parse_args()
    
    main(args)
