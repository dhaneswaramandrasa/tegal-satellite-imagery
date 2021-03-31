
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image 
import os
import geopandas as gpd
import sys
import time
from datetime import timedelta
from tqdm import tqdm
from time import sleep
from torch.autograd import Variable
import time

def define_model():
    model = models.resnet50(pretrained=True)

        # check if CUDA is available
    use_cuda = torch.cuda.is_available()
            
        # stop gradient updates for all parameters
    for param in model.parameters():
        param.requires_grad = False
            
        # create new classifier for the dog breed problem (last output layer)
    model.fc = nn.Linear(2048, 10, bias=True)

        # set the gradient to update for the classifier layer 
    for param in model.fc.parameters():
        param.requires_grad = True
            
        # print the model
        # model_transfer
    if use_cuda:
        model = model.cuda()

    return model

def predict_satellite_transfer(img_path, model, model_filepath):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # load the image and return the predicted breed
    image_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                     normalize])

    image = Image.open(img_path).convert('RGB')
    image_tensor = image_transform(image)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        image_tensor = image_tensor.cuda()

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = Variable(image_tensor)

    model.load_state_dict(torch.load(model_filepath))
    model.eval()

    output = model(image_tensor)
    output = output.cpu()
    #index = output.argmax().item()
    index = output.data.numpy().argmax()

    #class_list = os.listdir("label/")
    class_list = ['building','constructed land','crops','housing complex','meadow','residential area','river','road','sea lake','tree cover']
    class_list = sorted(class_list)
    #class_list = class_list[1:]
    #class_names = [item[4:].replace("_", " ") for item in class_list]

    return class_list[index]
    #return index

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def create_satellite_label(model, box_filepath, model_filepath, grid_output):
    grid = gpd.read_file(box_filepath)
    #grid = grid.rename(columns={'id_left':'id'})
    grid['label'] = 'unknown'

    
    for i in tqdm(range(grid.shape[0])):
        gid = grid['gid_100'][i]

        filepath = "small_tiff_images/" + str(gid) + ".tiff" 

        grid.loc[grid.gid_100 == gid, 'label'] = predict_satellite_transfer(filepath, model, model_filepath)

        sleep(0)
        
    grid.to_file(grid_output)

def main():
    """
    Loading the data from csv files, merging the data, removing duplicate data, and saving it in SQL database.
    
    """
    
    if len(sys.argv) == 4:

        model_filepath, box_filepath, grid_output = sys.argv[1:]

        start_time = time.monotonic()
        print('Define model...')
        model = define_model()

        print('Create label...')
        create_satellite_label(model, box_filepath, model_filepath, grid_output)
        
        print('Finish!')

        end_time = time.monotonic()
        print('Duration: {}'.format(timedelta(seconds=end_time - start_time)))
    
    else:
        print('Check again!')

if __name__ == '__main__':
    main()