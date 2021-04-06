from PIL import Image, ImageDraw
import json
import os

def label_image(im, preds, width=2):
    draw = ImageDraw.Draw(im)
    for pred in preds:
        draw.rectangle([int(pred[2]), int(pred[3]), int(pred[0]),
                        int(pred[1])], outline='red', width=width)
    return im

if __name__=="__main__":

    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # set a path for saving predictions:
    preds_path = '../data/hw01_preds'

    # get sorted list of files:
    file_names = sorted(os.listdir(data_path))

    # remove any non-JPEG files:
    file_names = [f for f in file_names if '.jpg' in f]

    # load preds
    with open(os.path.join(preds_path, 'preds.json'), 'r') as f:
        preds = json.load(f)

    print(preds)

