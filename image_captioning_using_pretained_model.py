
import os
import pickle
import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, ViTModel, AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.cuda.amp import GradScaler, autocast
import random

folder_path = './image_caption/'
# Define paths to CSV files and image folders
train_csv_file = folder_path + 'train.csv'
train_img_folder = folder_path + 'train'
test_csv_file = folder_path + 'test.csv'
test_img_folder = folder_path + 'test'
val_csv_file = folder_path + 'val.csv'
val_img_folder = folder_path + 'val'

def get_preprocessed_caption(caption):
    caption = caption.replace('\s+', ' ')
    caption = caption.replace('[^A-Za-z]', '')
    caption = caption.strip().lower()
    caption = "<START> " + caption + " <END>"
    return caption

def load_caption(csv_file, img_folder):
    # Load CSV file
    data = pd.read_csv(csv_file)

    # Initialize lists to store image paths and corresponding captions
    captions = []
    img_dict = {}

    # Iterate over each row in the CSV file
    for index, row in data.iterrows():
        # Construct the image file path
        image_path = os.path.join(img_folder, row['filename'])
        image_name = image_path.split('/')[-1].split('.')[0]
        # Append the image path and caption to the lists
        caption = get_preprocessed_caption(row['caption'])
        captions.append(caption)
        img_dict[image_name] = caption

    return img_dict

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Configure logging to overwrite the log file
logging.basicConfig(
    filename='training_log.txt',  # Log file name
    filemode='w',                 # Overwrite the log file
    level=logging.INFO,           # Minimum log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

class CustomCaptionsDataset(Dataset):
    def __init__(self, root_dir, csv_file, preprocess_caption, transform=None, chunk_size=3):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.preprocess_caption = preprocess_caption
        self.chunk_size = chunk_size

        # Prepare a new list to hold the split data
        self.image_caption_pairs = []

        print("data preprocessing")
        # Process each image-caption pair
        for idx in tqdm(range(len(self.data[:]))):
            img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]))
            image = Image.open(img_name).convert('RGB')
            caption = self.data.iloc[idx, 2]

            # Split the caption into sentences
            sentences = caption.split('. ')
            sentences = [sentence.strip() + '.' for sentence in sentences if sentence]

            # Create overlapping chunks
            chunks = []
            for i in range(0, len(sentences), self.chunk_size - 1):
                chunk = sentences[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size and i != 0:
                    chunk = sentences[-self.chunk_size:]
                chunks.append(' '.join(chunk))

            # Create multiple entries for each chunk
            for chunk in chunks:
                processed_caption = self.preprocess_caption(chunk)
                self.image_caption_pairs.append((img_name, processed_caption))

        # Shuffle the data to ensure diversity
        random.shuffle(self.image_caption_pairs)

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        img_name, caption = self.image_caption_pairs[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption, img_name

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a common size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])



# Create dataset instances
train_dataset = CustomCaptionsDataset(root_dir=train_img_folder, csv_file=train_csv_file, preprocess_caption=get_preprocessed_caption, transform=transform)
test_dataset = CustomCaptionsDataset(root_dir=test_img_folder, csv_file=test_csv_file, preprocess_caption=get_preprocessed_caption, transform=transform)
val_dataset = CustomCaptionsDataset(root_dir=val_img_folder, csv_file=val_csv_file, preprocess_caption=get_preprocessed_caption, transform=transform)

# Tokenize captions
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {'bos_token': '<START>', 'eos_token': '<END>', 'pad_token': '<PAD>'}
tokenizer.add_special_tokens(special_tokens)
# Print vocab size
print("Vocabulary size:", len(tokenizer))

def collate_fn(batch):
    images, captions, idx = zip(*batch)
    images = torch.stack(images, 0)

    captions = list(captions)

    # Tokenize captions
    tokenized_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_captions['input_ids']
    attention_mask = tokenized_captions['attention_mask']

    # Return images, input_ids, and attention_mask for each batch
    return images, input_ids, attention_mask, idx

# DataLoader
batch_size = 16
batch_test_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)


from transformers import ViTModel, GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2Config, GPT2LMHeadModel
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import csv

# Vision Transformer Encoder
class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "google/vit-base-patch16-224"):
        super(ViTEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state[:, 0, :]

# GPT-2 Decoder
class GPT2Decoder(nn.Module):
    def __init__(self, encoder_dim: int, pretrained_model_name: str = "gpt2"):
        super(GPT2Decoder, self).__init__()
        self.tokenizer = tokenizer  # Use the previously defined tokenizer
        self.gpt2_config = GPT2Config.from_pretrained(pretrained_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name, config=self.gpt2_config)

        # Adjust GPT2's embedding layer to include additional special tokens
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

        # Retrieve special token IDs
        self.start_token_id = self.tokenizer.convert_tokens_to_ids('<START>')
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids('<PAD>')
        self.end_token_id = self.tokenizer.convert_tokens_to_ids('<END>')

        # Ensure the token IDs are properly set
        assert self.start_token_id == self.tokenizer.convert_tokens_to_ids('<START>'), "Start token ID mismatch"
        assert self.pad_token_id == self.tokenizer.pad_token_id, "Pad token ID mismatch"
        assert self.end_token_id == self.tokenizer.convert_tokens_to_ids('<END>'), "End token ID mismatch"
        
        # Create linear layer to map image embeddings to GPT2's hidden size
        self.image_embedding = nn.Linear(encoder_dim, self.gpt2_config.n_embd)

    def forward(self, encoder_hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Apply image embedding layer
        image_embeds = self.image_embedding(encoder_hidden_states).unsqueeze(1)

        # Extend attention mask to include image token
        if attention_mask is not None:
            extended_attention_mask = torch.cat([
                torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype),  # Mask for image token
                attention_mask  # Mask for text tokens
            ], dim=1)
        else:
            # If no attention_mask is provided, create a mask with all ones
            extended_attention_mask = torch.ones((input_ids.size(0), image_embeds.size(1) + input_ids.size(1)), device=image_embeds.device, dtype=torch.float)

        # Compute input embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        # Combine image embeddings and input embeddings
        combined_embeds = torch.cat((image_embeds, inputs_embeds), dim=1)

        # Forward pass through GPT-2 model
        outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=extended_attention_mask)

        return outputs.logits


# Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ViTEncoder()
        encoder_dim = self.encoder.vit.config.hidden_size
        self.decoder = GPT2Decoder(encoder_dim=encoder_dim)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        encoder_hidden_states = self.encoder(images)
        logits = self.decoder(encoder_hidden_states, input_ids, attention_mask)
        return logits

# Initialize Model
model = ImageCaptioningModel()

def tensor_to_pil_image(tensor, device):
    tensor = tensor.clone().detach().to(device)  # Ensure the tensor is on the correct device
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    tensor = tensor.clamp(0, 1)  # Clamp to ensure the values are in the correct range
    return transforms.ToPILImage()(tensor.cpu())  # Convert to PIL image, move to CPU for ToPILImage()

def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> float:
    model.train()
    model.to(device)
    total_loss = 0
    scaler = GradScaler()
    log_total_loss = 0
    log_avg_loss = 0

    for batch_idx, (images, input_ids, attention_mask, img_name) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        max_length = input_ids.shape[1]
        batch_loss = 0

        for i in range(1, max_length - 1, 2):
            with autocast():
                # Slice input_ids and attention_mask for the current sequence length
                outputs = model(images, input_ids[:, :i], attention_mask[:, :i])
                
                # Remove the last token from outputs (for predicting the next token)
                outputs = outputs[:, :-1, :].contiguous()
                
                # Flatten outputs to [batch_size * (sequence_length - 1), vocab_size]
                outputs_flat = outputs.view(-1, outputs.size(-1))
                
                # Prepare target tensor
                targets = input_ids[:, 1:i+1].reshape(-1)
                
                # Calculate loss
                loss = loss_fn(outputs_flat, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss += loss.item()
        
        avg_batch_loss = batch_loss 
        total_loss += batch_loss
        log_total_loss += batch_loss
        log_avg_loss += (batch_loss / ((max_length - 2) /2))
        
        if (batch_idx + 1) % 100 == 0:
            logging.info(f'Batch {batch_idx + 1}, Loss: {log_total_loss:.4f}, AVG Batch Loss: {log_total_loss/100:.4f}, AVG Run Loss: {log_avg_loss/100:.4f}')
            log_total_loss = 0
            log_avg_loss = 0

    average_loss = total_loss / len(dataloader)
    return average_loss

def compute_cider_score(actual, predicted):
    cider_scorer = Cider()
    score , cider_scores = cider_scorer.compute_score(actual, predicted)
    return score, cider_scores # 1.2

def compute_spice_score(actual, predicted):
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(actual, predicted)
    return spice_score  # 0.4

def compute_rouge_l_score(actual, predicted):
    rouge = Rouge()
    scores = rouge.get_scores(predicted, actual, avg=True)
    return scores['rouge-l']['f']  # 0.4 - 0.5

# Function to load the model
def load_model(model_class, model_path, device):
    # Initialize the model architecture
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model

def evaluate_model(model, dataloader, device, tokenizer, max_length=60, num_samples=5):
    model.eval()  # Set the model to evaluation mode
    generated_captions = []
    actual_captions = []
    img_names = []

    # Convert dataloader to a list to allow indexing
    dataloader_list = list(dataloader)
    random_indices = random.sample(range(len(dataloader_list)), num_samples)
    
    with torch.no_grad():  # No need to track gradients
        for i in tqdm(random_indices):
            images, cap, _, img_name = dataloader_list[i]
            images = images.to(device)
            
            input_ids = torch.full((images.size(0), 1), tokenizer.encode(tokenizer.bos_token)[0], dtype=torch.long).to(device)
            for _ in range(max_length):
                outputs = model(images, input_ids)
                predictions = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat((input_ids, predictions), dim=-1)
                if torch.all(predictions.squeeze(-1) == tokenizer.encode(tokenizer.eos_token)[0]):
                    break

            # Decode the generated ids to text
            for ids in input_ids:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                generated_captions.append(caption)
            
            for ids in cap:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                actual_captions.append(caption)
            
            for ids in img_name:
                img_names.append(ids)

    return compute_rouge_l_score(actual_captions, generated_captions)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

try: 
    model = load_model(ImageCaptioningModel, 'working_model_v2.pth', device)
    print("loading a previous model")
except:
    print("no model found, training a new model")

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training parameters
num_epochs = 10
best_val_score = float('-inf')
best_model_path = 'best_model_v2.pth'

# Training loop
for epoch in tqdm(range(num_epochs)):
    print(f'Starting Epoch {epoch + 1}/{num_epochs}')

    # Train the model
    avg_loss = train(model, train_dataloader, optimizer, loss_fn, device)
    print(f'Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}')

    final_model_path = 'working_model_v2.pth'
    torch.save(model.state_dict(), final_model_path)

    # Validate the model
    val_score = evaluate_model(model, val_dataloader, device, tokenizer)
    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Score: {val_score:.4f}')

    working_model_path = 'working_model_v2.pth'
    torch.save(model.state_dict(), working_model_path)
    # Save the best model
    if val_score >= best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with validation score: {best_val_score:.4f}')

# Save final model
final_model_path = 'final_model_v2.pth'
torch.save(model.state_dict(), final_model_path)
print(f'Saved final model to: {final_model_path}')


# model_path = 'best_model_v2.pth'  # Path to the saved model

# Assuming `ImageCaptioningModel` is the class of your model
# model_trained = load_model(ImageCaptioningModel, model_path, device)


def beam_search(model, images, tokenizer, device, beam_width=5, max_length=60):
    start_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.eos_token_id

    # Initial token for each candidate sequence
    initial_input_ids = torch.full((images.size(0), 1), start_token_id, dtype=torch.long).to(device)

    # Encode images once
    encoder_hidden_states = model.encoder(images)

    # Initialize beam candidates: (sequence, log probability)
    candidates = [(initial_input_ids, torch.zeros(images.size(0), device=device))]  # log_prob is now a tensor

    # Beam search loop
    for _ in range(max_length):
        new_candidates = []

        for candidate_input_ids, log_prob in candidates:
            # Get the logits for the last predicted token
            outputs = model.decoder(encoder_hidden_states, candidate_input_ids)
            logits = outputs[:, -1, :]

            # Apply log_softmax to convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the top K candidates for each sequence
            topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                # Update the sequence with the new token
                new_input_ids = torch.cat((candidate_input_ids, topk_indices[:, i:i+1]), dim=-1)

                # Calculate the new log probability
                new_log_prob = log_prob + topk_probs[:, i]

                # Add the new candidate to the list
                new_candidates.append((new_input_ids, new_log_prob))

        # Select top K candidates based on their log probabilities
        new_candidates.sort(key=lambda x: x[1].sum().item(), reverse=True)  # Sort by the sum of log probabilities
        candidates = new_candidates[:beam_width]

        # Check if all candidates have ended with the end token
        all_end_tokens = all(torch.all(candidate[0][:, -1] == end_token_id) for candidate in candidates)
        if all_end_tokens:
            break

    # Return the top candidate sequence for each batch element
    top_sequences = [candidates[0][0][i] for i in range(images.size(0))]
    return top_sequences


def evaluate_model_beam_random(model, dataloader, device, tokenizer, max_length=60, num_samples=4, beam_size=3):
    model.eval()  # Set the model to evaluation mode
    generated_captions = []
    actual_captions = []
    count = 0

    dataloader_iter = iter(dataloader)
    random_indices = random.sample(range(len(dataloader)), num_samples)

    # Truncate the file before writing new captions
    output_file_path = "./output_v2_val.csv"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    # Convert dataloader to a list to allow indexing
    dataloader_list = list(dataloader)
    random_indices = random.sample(range(len(dataloader_list)), num_samples)
    
    with torch.no_grad():  # No need to track gradients
        for i in tqdm(random_indices):
            images, original_caption, _, img_name = dataloader_list[i]
            images = images.to(device)
            # Generate captions using beam search
            generated_ids = beam_search(model, images, tokenizer, device, beam_width=beam_size, max_length=max_length)

            # Decode the generated ids to text
            for ids in generated_ids:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                generated_captions.append(caption)
            
            for ids in original_caption:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                actual_captions.append(caption)
            
            # Decode the generated sequence
            for ids, cap, name in zip(generated_ids, original_caption, img_name):
                pred_caption = tokenizer.decode(ids, skip_special_tokens=True)
                original = tokenizer.decode(cap, skip_special_tokens=True)
                print("\ncount - ", count)
                print("original caption - ", original)
                print("image_name - ", name)
                print("predicted caption - ", pred_caption)
                with open(output_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([count, name, pred_caption, original])
                    count += 1

    return compute_rouge_l_score(actual_captions, generated_captions)

def evaluate_model_beam(model, dataloader, device, tokenizer, max_length=60, beam_size=3):
    model.eval()  # Set the model to evaluation mode
    generated_captions = []
    actual_captions = []
    count = 0

    # Truncate the file before writing new captions
    output_file_path = "./output_v2.csv"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    with torch.no_grad():  # No need to track gradients
        for images, original_caption, _, img_name in tqdm(dataloader):
            images = images.to(device)
            # Generate captions using beam search
            generated_ids = beam_search(model, images, tokenizer, device, beam_width=beam_size, max_length=max_length)

            # Decode the generated ids to text
            for ids in generated_ids:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                generated_captions.append(caption)
            
            for ids in original_caption:
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                actual_captions.append(caption)
            
            # Decode the generated sequence
            for ids, cap, name in zip(generated_ids, original_caption, img_name):
                pred_caption = tokenizer.decode(ids, skip_special_tokens=True)
                original = tokenizer.decode(cap, skip_special_tokens=True)
                print("\ncount - ", count)
                print("original caption - ", original)
                print("image_name - ", name)
                print("predicted caption - ", pred_caption)
                with open(output_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([count, name, pred_caption, original])
                    count += 1

    return compute_rouge_l_score(actual_captions, generated_captions)

val_score = evaluate_model_beam_random(model, val_dataloader, device, tokenizer, beam_size=3)
print("val score - ",val_score)
test_score = evaluate_model_beam(model, test_dataloader, device, tokenizer, beam_size=3)
print("test score - ",test_score)

pred_caption = []  # List to store the loaded data
actual_caption=[]
actual_dict = dict()
predicted_dict = dict()

with open( "./output/output.csv", mode='r', newline='') as file:
    reader = csv.reader(file)
    for idx, img_name, pred, original in tqdm(reader):
        pred_caption.append(pred)
        actual_caption.append(original)

        if img_name not in actual_dict.keys():
            actual_dict[img_name] = []
            predicted_dict[img_name] = []
        actual_dict[img_name].append(original)
        predicted_dict[img_name].append(pred)

rouge_l_score = compute_rouge_l_score(actual_caption, pred_caption)
print(f"\nROUGE-L: {rouge_l_score}")

cider_score = compute_cider_score(actual_dict, predicted_dict)
print(f"CIDEr: {cider_score}")

spice_score = compute_spice_score(actual_dict, predicted_dict)
print(f"SPICE: {spice_score}")

