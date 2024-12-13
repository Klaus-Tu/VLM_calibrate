import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_fea(loader, model, device):
    model.eval()

    fea_list = []
    labels_list = []

    for j, (input, label) in enumerate(tqdm(loader)):
        with torch.no_grad():
            input = input.to(device)
            label = label.to(device)

            image_features = model.encode_image(input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        fea_list.append(image_features)
        labels_list.append(label)

    features = torch.cat(fea_list)
    labels = torch.cat(labels_list)

    return features.cpu(), labels.cpu()


def zeroshot_classifier(model, tokenizer, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in (classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights
