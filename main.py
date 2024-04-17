import os
import pandas as pd
from PIL import Image
import re
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import threading
from collections import Counter
import pickle

from visualization import visualize_samples, display_results
from model import Seq2SeqAttentionCNN
from training import model_training, feedforward

# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'



# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
        self.add_mapping('<pad>', 0)
        self.add_mapping('<sos>', 1)
        self.add_mapping('<eos>', 2)
        self.add_mapping('<unk>', 3)
    
    def __len__(self):
        return len(self.key_to_value)
    
    def keys(self):
        return self.key_to_value.keys()
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key, 3)

    def get_key(self, value):
        return self.value_to_key.get(value, '<unk>')
    
    
    
# data structure to store image and caption
class Data():
    def __init__(self, name):
        self.name = name
        self.caption = None
        self.token = None
        self.img = None
    
    def set_image(self, img):
        self.img = img
        
    def set_caption(self, caption):
        self.caption = caption
    
    def set_token(self, token):
        self.token = token
    
    def get_image(self):
        return self.img
    
    def get_caption(self):
        return self.caption
    
    def get_name(self):
        return self.name
    
    def get_token(self):
        return self.token
    

# model inference
def inference(model, X, Y, device, max_length):
    with torch.no_grad():
        # Set the encoder to evaluation mode
        model.eval()

        # Move input tensor to device
        X = X.to(device)
        
        # inference
        pred = model.inference(X, max_length)
        decoded_words = tensorToTokens(Y, pred)
        return decoded_words
    
    

# pad image to square
class PadToSquare(torch.nn.Module):
    def __call__(self, img):
        # Calculate the padding needed to make the image square
        width, height = img.size
        max_dim = max(width, height)
        pad_width = max_dim - width
        pad_height = max_dim - height

        # Apply padding to make the image square
        padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
        padded_img = transforms.functional.pad(img, padding, fill=0)

        return padded_img
    
    
        
# reading image and captions from folder
def read_labels(filename):
    df = pd.read_table(filename, sep='|')
    
    # strip the column names
    for name in df.columns.tolist():
        df = df.rename(columns={name:name.strip()})
    
    # filter out the invalid row
    df.dropna(subset=['comment'], inplace=True)
    
    # combine the comment_number and comment column
    df = df.groupby('image_name')['comment'].apply(lambda group: group.values.tolist()).reset_index(name='comments')
    
    # convert pandas dataframe to list(Data)
    data = []
    for index in range(len(df)):
        row = df.iloc[index]
        element = Data(row['image_name'])
        element.set_caption(row['comments'])
        data.append(element)        
    return data
    

# reading image using parallel processing
def parallel_read_images(img_path, data, transform, num_threads=8):
    
    def read_image(img_path, data, transform, start_idx, end_idx, progress_bar):
        for i in range(start_idx, end_idx):
            img = Image.open(os.path.join(img_path, data[i].get_name())).convert('RGB')
            img = transform(img)  # resize transform
            data[i].set_image(img)
            progress_bar.update(1)  # Update the progress bar
            
    
    # Split the data indices into equal parts for each thread
    chunk_size = len(data) // num_threads
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    chunks[-1] = (chunks[-1][0], len(data))  # Adjust the last chunk to include remaining indices

    progress_bar = tqdm(total=len(data), desc='Processing Images', unit='image')
    threads = []
    for start_idx, end_idx in chunks:
        thread = threading.Thread(target=read_image, args=(img_path, data, transform, start_idx, end_idx, progress_bar))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Close the progress bar
    progress_bar.close()







# english tokenizer
def en_tokenizer(text):
    # Define regex pattern to match words
    pattern = r"\w+|[^\w\s]"
    tokens = re.findall(pattern, text)
    tokens = [token.lower() for token in tokens] # convert to lower case
    return tokens



# build word to ind dictionary
def build_vocab(data, min_freq=1):
    # Initialize an empty Counter object to count word frequencies
    word_counts = Counter()
    
    # Count word frequencies in all sentences
    for element in data:
        tokens = element.get_token()
        word_counts.update(tokens)
        
                
    # Create a vocabulary mapping from words to indices
    vocab = BidirectionalMap()
    
    for word, freq in word_counts.items():
        # ignore the word count with too few frequency
        if freq < min_freq: 
            continue
        if word not in vocab.keys():
            vocab.add_mapping(word, len(vocab))
           
    return vocab


# convert from a sentence to a torch tensor
def tokensToTensor(lang, sentence, max_length):
    # Tokenize sentence and convert tokens to indices using the vocabulary
    tokens = [lang.get_value(token) for token in sentence]
    tokens = tokens[:max_length - 1]  # Truncate sentence if it's longer than max_length - 1
    
    # tokens input has <sos> in the front
    tokens_input = tokens.copy()
    tokens_input.insert(0, lang.get_value('<sos>'))
    tokens_input += [lang.get_value('<pad>')] * (max_length - len(tokens_input))
    tokens_input = torch.tensor(tokens_input, dtype=torch.long)
    
    # tokens output has <eos> at the end
    tokens_output = tokens.copy()
    tokens_output.append(lang.get_value('<eos>'))  # Append end-of-sequence token
    tokens_output += [lang.get_value('<pad>')] * (max_length - len(tokens_output))
    tokens_output = torch.tensor(tokens_output, dtype=torch.long)

    return tokens_input, tokens_output
    


# convert from a torch tensor to sentence
def tensorToTokens(lang, tensor):
    # Convert tensor to list
    tokens = tensor
    
    # replace the last ind with eos (incase eos is not in sentence)
    tokens[-1] = lang.get_value('<eos>')
    
    # Find the index of the end-of-sequence token
    eos_index = tokens.index(lang.get_value('<eos>'))
    # Remove padding tokens and end-of-sequence token
    tokens = tokens[:eos_index]
    
    # Convert indices back to tokens
    tokens = [lang.get_key(token) for token in tokens]
    return tokens



def dataloader(X, Y, batch_size):    
    # Combine source and target sentences into tuples
    dataset = list(zip(X, Y))
    
    # Create DataLoader with zipped dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader




def main():
    # model image resize transform
    resize_transform = transforms.Compose([
        PadToSquare(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = '../Datasets/flickr30k/'
    # reading labels into pandas df
    data = read_labels(data_dir + 'results.csv')
    
    # reading images
    img_path = data_dir + 'flickr30k_images'
    parallel_read_images(img_path, data, resize_transform)
    
    # visualize examples
    for i in range(0, 500, 100):
        visualize_samples(data[i])
    

    # tokenize sentences
    for i in range(len(data)):
        # there are several comments for a single image, only keep one. 
        # can be modified in data augmentation step
        comments = data[i].get_caption()
        data[i].set_caption(comments[0])
        
        # tokenize
        tokenized = en_tokenizer(data[i].get_caption())
        data[i].set_token(tokenized)
    

    # building the vocabulary
    if 'English.pkl' in os.listdir():
        with open('English.pkl', 'rb') as f:    
            vocab = pickle.load(f)
    else:
        vocab = build_vocab(data)
        with open('English.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    
    
    # token to Tensor
    max_length = 64
    for i in range(len(data)):
        tokens = data[i].get_token()
        tokens = tokensToTensor(vocab, tokens, max_length)[1]
        data[i].set_token(tokens)
    print('done convert to tensor')
    print('data', len(data))
    
    # train, valid, and test split
    train = data[:int(0.8*len(data))]
    valid = data[int(0.8*len(data)):int(0.9*len(data))]
    test = data[int(0.9*len(data)):]
    trainX, trainY = [x.get_image() for x in train], [x.get_token() for x in train]
    validX, validY = [x.get_image() for x in valid], [x.get_token() for x in valid]
    testX, testY = [x.get_image() for x in test], [x.get_token() for x in test]
    del data, train, valid, test
    print('done train test split')
    
    
    # create train and valid data loader
    batch_size = 16
    train_dataloader = dataloader(trainX, trainY, batch_size)
    valid_dataloader = dataloader(validX, validY, batch_size)
    del trainX, trainY, validX, validY
    print('done created loader')
    
    # loading model
    output_dim = len(vocab)
    sos_token = vocab.get_value('<sos>')
    eos_token = vocab.get_value('<eos>')
    
    model = Seq2SeqAttentionCNN(output_dim, sos_token, eos_token)
    model = model.to(GPU_Device()) # move the model to GPU

    if f'{type(model).__name__}.pth' in os.listdir():
        model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # model training
    model_training(train_dataloader, valid_dataloader, model, GPU_Device())

    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    
    # load the test dataset
    test_dataloader = dataloader(testX, testY, batch_size)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_blue = feedforward(test_dataloader, model, criterion, GPU_Device())
    print(f'Test BLUE: {test_blue:.3f} | Test Loss: {test_loss:.3f}')
    
    for i in range(0, 50, 10):
        element = testX[i]
        ground_truth = tensorToTokens(vocab, testY[i].tolist())
        predY = inference(model, element, vocab, GPU_Device(), max_length)
        sentencePred = ' '.join(predY)
        sentenceY = ' '.join(ground_truth)
        display_results(element, sentenceY, sentencePred)
    


if __name__ == '__main__':
    main()