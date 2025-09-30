'''Automatically load and preprocess various multimodal datasets for training a model'''
import os
import cv2
import yaml
import numpy as np
import torch
import base64
import hashlib
import json
import types
import urllib
import io
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from torch import nn
from datasets import load_dataset
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tokenizers.pre_tokenizers import BertPreTokenizer, WhitespaceSplit
from PIL import Image, ImageDraw, ImageFont
from datasets.features import Image as ImageFeature, ClassLabel, Value
from datasets import Dataset as HFDataset


###########################################################################################
# Utils
###########################################################################################

def tokens_to_frames(tokens, frame_size=64, sequence_length=None, img_type='L', padding_side="left", use_special_padding_frame=False):
    '''Convert a list of text tokens to a list of frames'''

    # If sequence length is specified, pad or truncate the tokens
    if sequence_length is not None:
        if len(tokens) < sequence_length:
            if padding_side == "left":
                tokens = ['<pad>'] * (sequence_length - len(tokens)) + tokens
            elif padding_side == "right":
                tokens = tokens + ['<pad>'] * (sequence_length - len(tokens))
            else:
                raise ValueError("padding_side must be 'left' or 'right'")
        else:
            tokens = tokens[-sequence_length:]

    frames = []
    for token in tokens:
        if token == "<pad>" and use_special_padding_frame:
            # Use frame filled with -1 for padding instead of a real frame
            # This is useful for ignoring padding frames in the loss function
            frame = torch.ones((frame_size, frame_size), dtype=torch.float32) * -1
            frame = frame.squeeze(0)
            # add 3 channel dimension
            frame = frame.unsqueeze(0).expand(3, -1, -1)
            frames.append(frame)
        else:
            img = Image.new(img_type, (frame_size, frame_size), color=(0))
            font_height = min(frame_size/len(token) * 1.65, frame_size * 1) # Scale font size to fit
            font_height = max(font_height, 2) # Ensure font height is positive
            fnt = ImageFont.truetype('./data/FreeMono.ttf', font_height) # Load a monospace font
            d = ImageDraw.Draw(img)
            d.fontmode = "L" # Anti-aliasing
            d.fill = "white"
            d.text((0,frame_size/2 - font_height/2), token, font=fnt, fill="white")
            frame = pil_to_tensor(img)
            frame = frame.squeeze(0)
            frames.append(frame)
    frames = torch.stack(frames, dim=0)
    return frames


###########################################################################################
# Wrapper Classes
###########################################################################################

class CLM_Wrapper(Dataset):
    '''Wrapper for a dataset that returns a single frame at a time for training a CLM'''
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def vocab(self):
        '''Returns target vocab'''
        return self.dataset.vocab()

    def __getitem__(self, idx):
        '''Return a single frame at a time'''
        item = self.dataset[idx]

        if item['targets'] is None:
            return item

        return {**item, **{
            'inputs': torch.cat([item['inputs'], item['targets']], dim=0),
            'targets': None,
        }}


class Augmentations_Wrapper(Dataset):
    '''Apply augmentations to a dataset'''
    def __init__(self, dataset, frame_idxs_to_augment):
        self.dataset = dataset
        self.augmentations = transforms.Compose([
            transforms.Pad(8, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=self.dataset.frame_size, padding=8, padding_mode='reflect'),
            # transforms.RandomRotation(degrees=5),
            # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.frame_idxs_to_augment = frame_idxs_to_augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''Apply augmentations to the specified frames'''
        item = self.dataset[idx]
        frames = []
        augmented_frames = []
        for i, frame in enumerate(item['inputs']):
            div = 255 if torch.max(frame) > 1 else 1
            frame = frame / div
            frames.append(frame)

        all_frames = torch.cat([frame.unsqueeze(0) for frame in frames], dim=0)
        all_frames = self.augmentations(all_frames)

        for i in range(len(frames)):
            if i in self.frame_idxs_to_augment:
                augmented_frames.append(torch.round(all_frames[i] * div))
            else:
                augmented_frames.append(item['inputs'][i])
        item['inputs'] = torch.stack(augmented_frames, dim=0)
        return item


class Cache_Wrapper(Dataset):
    '''Wrapper for a dataset that caches the results of __getitem__ as npz files'''
    def __init__(self, dataset):
        self.dataset = dataset
        self.printed = False
        self.frame_size = self.dataset.frame_size
        # try to get out frames from dataset
        try:
            output_frames = self.dataset.output_frames
        except AttributeError:
            output_frames = 1

        self.cache_dir = f'cache/{self.dataset.__class__.__name__}_{self.dataset.split}_{self.dataset.input_frames}_{output_frames}_{self.__hash_dataset()}'
        print(f"Caching to {self.cache_dir}")

        if not os.path.exists('cache'):
            os.makedirs('cache')

    def __len__(self):
        return len(self.dataset)

    def __hash_dataset(self):
        '''Hash the dataset for caching purposes'''
        # Hash anything that's a simple value (not a list or dict)
        hashable = {key: value for key, value in self.dataset.__dict__.items() if isinstance(value, (int, str, float, bool)) and key not in ["sample_frac"]}
        return hashlib.sha256(str(hashable).encode()).hexdigest()[:8]

    def vocab(self):
        '''Returns target vocab'''
        return self.dataset.vocab()

    def __getitem__(self, idx):
        if not os.path.exists(f"{self.cache_dir}_{idx}"):
            item = self.dataset[idx]
            TensorDict(item).save(f"{self.cache_dir}_{idx}")
        else:
            if self.printed is False:
                # print("Loading from cache")
                self.printed = True

            item = TensorDict.load(f"{self.cache_dir}_{idx}")
        return item


class Multitask_Wrapper(Dataset):
    '''Wrapper for multiple datasets that samples from each dataset with equal probability'''
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_weights = [1/len(datasets) for _ in datasets]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def vocab(self):
        '''Returns target vocab'''
        return None

    def __getitem__(self, idx):
        '''Sample from each dataset with equal probability'''
        dataset_idx = np.random.choice(len(self.datasets), p=self.dataset_weights)

        # Add frame to indicate which dataset the sample came from
        dataset_frame = tokens_to_frames(
            [self.datasets[dataset_idx].__class__.__name__],
            frame_size = self.datasets[dataset_idx].frame_size,
            img_type = self.datasets[dataset_idx].img_type
        )
        item = self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])]
        item['inputs'] = torch.cat([dataset_frame, item['inputs']], dim=0)
        item["dataset"] = self.datasets[dataset_idx].__class__.__name__
        return item



###########################################################################################
# Dataset Classes
###########################################################################################

class GLUE(Dataset):
    '''Renders GLUE benchmark tasks as a video sequence'''
    def __init__(self, subset, split, img_type='greyscale', input_frames=20, frame_size=64, sample_frac=1):

        if split == "val":
            split = "validation"

        if split == "test":
            print("WARNING - using validation split for test data")
            split = "validation"

        self.dataset = load_dataset('glue', subset)[split]
        self.tokenizer = WhitespaceSplit()
        self.subset = subset
        self.split = split
        self.input_frames = input_frames
        self.output_frames = 1
        self.sample_frac = sample_frac

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.input_frames = input_frames
        self.frame_size = frame_size

        # Rename columns
        self.dataset = self.dataset.rename_column('sentence', 'input')
        self.dataset = self.dataset.rename_column('label', 'output')

    def __len__(self):
        if self.sample_frac < 1:
            return int(len(self.dataset) * self.sample_frac)
        return len(self.dataset)

    def vocab(self):
        '''Returns target vocab'''
        return self.dataset.features['output'].names

    def __getitem__(self, idx):
        '''Encode the text as a video sequence'''
        input_text = self.dataset[idx % len(self)]['input'] + " |"
        input_tokens = self.tokenizer.pre_tokenize_str(input_text)
        input_tokens = [token[0] for token in input_tokens]
        frames_in = tokens_to_frames(
            input_tokens,
            sequence_length = self.input_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        output_text = self.dataset.features["output"].int2str(self.dataset[idx]['output'])
        output_tokens = self.tokenizer.pre_tokenize_str(output_text)
        output_tokens = [token[0] for token in output_tokens]
        frames_out = tokens_to_frames(
            output_tokens,
            sequence_length = 1,
            img_type = self.img_type,
            frame_size = self.frame_size,
            padding_side="right",
        )
        frames_in, frames_out = frames_in.float(), frames_out.float()
        # frames = torch.cat((frames_in, frames_out), 0)
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": input_text, "output": output_text}
        }


class ImageToText(Dataset):
    '''Renders image-to-text datasets as a video sequence'''
    def __init__(self, dataset_name, subset=None, split="train", img_type='L', input_frames=2, output_frames=1, frame_size=64, sample_frac=1, streaming=False):
        self.dataset = load_dataset(dataset_name, subset, streaming=streaming)

        # Standardize split names
        if split == "val":
            split = "validation"
        if split == "validation":
            if split not in self.dataset:
                split = "test"

        self.dataset = self.dataset[split]


        self.tokenizer = BertPreTokenizer()
        self.dataset_name = dataset_name
        self.sample_frac = sample_frac
        self.input_frames = input_frames
        self.output_frames = output_frames

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size

        # Standardize column names
        img_col = None
        text_col = None
        for col in self.dataset.features:
            if img_col is None and isinstance(self.dataset.features[col], ImageFeature):
                img_col = col
            elif text_col is None and isinstance(self.dataset.features[col], ClassLabel):
                text_col = col

        if img_col is None:
            # Find col containing url and load image from url
            for col in self.dataset.features:
                if isinstance(self.dataset.features[col], Value) and self.dataset.features[col].dtype == 'string':
                    if "url" in col:
                        img_col = col
                        break

        if text_col is None:
            for col in self.dataset.features:
                if isinstance(self.dataset.features[col], Value) and self.dataset.features[col].dtype == 'string' and col != img_col:
                    text_col = col
                    break


            if img_col is None:
                raise ValueError("No image column found in dataset")



        if img_col != 'image':
            self.dataset = self.dataset.rename_column(img_col, 'image')
        if text_col != 'label':
            self.dataset = self.dataset.rename_column(text_col, 'label')

        self.features = self.dataset.features.copy()
        self.length = self.dataset.info.splits[split].num_examples
        if streaming:
            self.dataset = iter(self.dataset)


    def __len__(self):
        return int(self.length * self.sample_frac)

    def vocab(self):
        """Returns target vocab"""
        if isinstance(self.features['label'], ClassLabel):
            return self.features['label'].names
        return None

    def _load_image_from_url(self, url):
        '''Load image from url'''
        try:
            with urllib.request.urlopen(url) as url:
                img = Image.open(io.BytesIO(url.read()))
            return img
        except Exception as e:
            return None

    def __getitem__(self, idx):
        '''Encode the image as a video sequence followed by the label'''''
        if isinstance(self.dataset, types.GeneratorType):
            # Streaming mode - just ignore idx, get next item
            item = next(self.dataset)

            # Load image from url if necessary
            if isinstance(item['image'], str):
                while True:
                    image = self._load_image_from_url(item['image'])
                    if image is not None:
                        break
                    item = next(self.dataset)
                item['image'] = image
        else:
            item = self.dataset[idx % len(self.dataset)]
            if isinstance(item['image'], str):
                item['image'] = self._load_image_from_url(item['image'])
                if item['image'] is None:
                    return None

        frames_in = [
            pil_to_tensor(item['image'].resize((self.frame_size, self.frame_size)).convert(self.img_type)).unsqueeze(0),
            tokens_to_frames(["|"], frame_size=self.frame_size, img_type=self.img_type)
        ]
        if self.input_frames > 2:
            frames_in = [tokens_to_frames([], frame_size=self.frame_size, img_type=self.img_type, sequence_length=self.input_frames-2)] + frames_in

        frames_in = torch.cat(frames_in, dim=0)

        if isinstance(self.features['label'], ClassLabel):
            if item["label"] == -1:
                output_text = "unknown"
            else:
                output_text = self.features['label'].int2str(item['label'])
        else:
            output_text = item['label'].lower()
        output_tokens = self.tokenizer.pre_tokenize_str(output_text)
        output_tokens = [token[0] for token in output_tokens]
        frames_out = tokens_to_frames(
            output_tokens,
            img_type = self.img_type,
            frame_size = self.frame_size,
            padding_side="right",
            sequence_length = self.output_frames
        )
        frames_in, frames_out = frames_in.float(), frames_out.float()
        # frames = torch.cat((frames_in, frames_out), 0)
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": "[IMG] |", "output": output_text}
        }


class CLEVRER(Dataset):
    '''Renders VQA tasks as a video sequence'''

    urls = {
            "train": {
                "videos": "http://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
                "qa": "http://data.csail.mit.edu/clevrer/questions/train.json"
            },
            "validation": {
                "videos": "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
                "qa": "http://data.csail.mit.edu/clevrer/questions/validation.json"
            },
    }

    def vocab(self):
        '''Returns target vocab'''
        return self._vocab

    def __init__(self, split="train", img_type='L', input_frames=2, output_frames=1, question_frames=20, frame_size=64, sample_frac=1):

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'

        if split == "val":
            split = "validation"
        if split == "test":
            print("WARNING - using validation split for test data")
            split = "validation"
        self.split = split

        self.frame_size = frame_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.question_frames = question_frames
        self.sample_frac = sample_frac
        self.tokenizer = BertPreTokenizer()

        # If dataset doesn't exist, download it
        if not os.path.exists('datasets/clevrer'):
            os.makedirs('datasets/clevrer')
        if not os.path.exists(f'datasets/clevrer/{split}.json'):
            os.system(f'wget {self.urls[split]["qa"]} -O datasets/clevrer/{split}.json')
        if not os.path.exists(f'datasets/clevrer/{split}'):
            os.system(f'wget {self.urls[split]["videos"]} -O datasets/clevrer/{split}.zip')
            os.system(f'unzip datasets/clevrer/{split}.zip -d datasets/clevrer/{split}')
            os.system(f'mv datasets/clevrer/{split}/*/* datasets/clevrer/{split}')
            # os.system(f'rm datasets/clevrer/{split}.zip')

        # Load Questions
        data = json.load(open(f'datasets/clevrer/{split}.json'))
        self.data = []
        self._vocab = []
        for video in data:
            for question in video['questions']:
                if "choices" not in question:
                    self.data.append({
                        'video_filename': video['video_filename'],
                        'question': question
                    })
                    self._vocab.extend(question['answer'].split())

        # Build vocab
        self._vocab += list([str(n) for n in range(10)])
        self._vocab += ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
        self._vocab += ['metal', 'rubber']
        self._vocab += ['sphere', 'cylinder', 'cube']
        self._vocab += ['yes', 'no']
        self._vocab = list(set(self._vocab))


    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def __getitem__(self, idx):
        '''Append video sequence to QA pair'''

        item = self.data[idx % len(self.data)]

        # Load video
        video_filename = item['video_filename']
        video = cv2.VideoCapture(f'datasets/clevrer/{self.split}/{video_filename}')
        vid_frames = []
        i = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if self.img_type == 'L':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif self.img_type.lower() == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Image type must be 'L' or 'RGB'")
            frame = Image.fromarray(frame)
            frame = frame.resize((self.frame_size, self.frame_size))
            frame = pil_to_tensor(frame)
            vid_frames.append(frame)

        # Subsample to fit in fixed number of frames
        num_vid_frames = self.input_frames - self.question_frames - 1
        subsample = len(vid_frames) // num_vid_frames
        vid_frames = vid_frames[::subsample]
        vid_frames = vid_frames[:num_vid_frames]
        num_vid_frames = len(vid_frames)
        try:
            vid_frames = torch.stack(vid_frames, dim=0)
        except:
            breakpoint()

        # Load question
        question = item['question']
        input_text = question['question'] + " |"
        input_tokens = self.tokenizer.pre_tokenize_str(input_text)
        input_tokens = [token[0] for token in input_tokens]
        frames_in = tokens_to_frames(
            input_tokens,
            sequence_length = self.input_frames - num_vid_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )
        frames_in = torch.cat([vid_frames, frames_in], dim=0)

        output_text = question['answer']
        output_tokens = self.tokenizer.pre_tokenize_str(output_text)
        output_tokens = [token[0] for token in output_tokens]
        frames_out = tokens_to_frames(
            output_tokens,
            sequence_length = self.output_frames,
            padding_side="right",
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        frames_in, frames_out = frames_in.float(), frames_out.float()
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": "[VIDEO] " + input_text, "output": output_text}
        }


class TinyVIRAT(Dataset):
    '''RendersTinyVIRAT: Low-resolution Video Action Recognition as a video classification task'''

    url = "https://www.crcv.ucf.edu/datasets/ugur/TinyVIRAT.zip"

    def __init__(self, split="train", img_type='L', input_frames=2, output_frames=1,frame_size=64, sample_frac=1):

        if split in ["val", "validation"]:
            split = "test"

        self.split = split
        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size
        self.tokenizer = BertPreTokenizer()
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sample_frac = sample_frac

        # If dataset doesn't exist, download it
        if not os.path.exists('datasets/tinyvirat'):
            os.makedirs('datasets/tinyvirat')
            os.system(f'wget {self.url} --no-check-certificate -O datasets/tinyvirat.zip')
            os.system(f'unzip datasets/tinyvirat.zip -d datasets/tinyvirat')
            os.system(f'mv datasets/tinyvirat/TinyVIRAT/* datasets/tinyvirat')
            os.system(f'rm -r datasets/tinyvirat/TinyVIRAT')
            os.system(f'rm datasets/tinyvirat.zip')

        # Load Questions
        self.data = json.load(open(f'datasets/tinyvirat/tiny_{split}.json'))["tubes"]

        # Build vocab
        self._vocab = []
        for item in self.data:
            labels = []
            for label in item['label']:
                labels += label.split("_")
            self._vocab += labels
        self._vocab = list(set(self._vocab))
        self._vocab.append(",")

    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def vocab(self):
        return self._vocab


    def __getitem__(self, idx):
        '''Construct video sequence from frames and append label'''

        item = self.data[idx % len(self.data)]

        # Load video
        video_filename = item['path']
        video = cv2.VideoCapture(f'datasets/tinyvirat/videos/{self.split}/{video_filename}')
        vid_frames = []


        while True:
            ret, frame = video.read()
            if not ret:
                break
            if self.img_type == 'L':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif self.img_type.upper() == 'RGB':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Image type must be 'L' or 'RGB'")
            frame = Image.fromarray(frame)
            frame = frame.resize((self.frame_size, self.frame_size))
            frame = pil_to_tensor(frame)
            vid_frames.append(frame)

        # Subsample to fit in fixed number of frames
        num_vid_frames = self.input_frames - 1
        subsample = len(vid_frames) // num_vid_frames
        vid_frames = vid_frames[::subsample]
        vid_frames = vid_frames[:num_vid_frames]
        num_vid_frames = len(vid_frames)
        # vid_frames = torch.stack(vid_frames, dim=0)
        vid_frames = vid_frames[(self.input_frames-1) * -1:]
        vid_frames = torch.stack(vid_frames, dim=0)

        # Add a separator token
        sep = tokens_to_frames(
            ["|"],
            sequence_length = 1,
            img_type = self.img_type,
            frame_size = self.frame_size
        )
        vid_frames = torch.cat([vid_frames, sep], dim=0)

        # Pad and truncate
        if vid_frames.shape[0] < self.input_frames:
            pad_frames = tokens_to_frames(
                [],
                sequence_length = self.input_frames - len(vid_frames),
                img_type = self.img_type,
                frame_size = self.frame_size
            ).squeeze(0)
            vid_frames = torch.cat([pad_frames, vid_frames], dim=0)

        output_text = ", ".join([label.replace("_", " ") for label in item['label']])
        output_tokens = []
        for label in item['label']:
            output_tokens += label.split("_")
            output_tokens.append(",")
        output_tokens = output_tokens[:-1] # Remove last comma
        frames_out = tokens_to_frames(
            output_tokens,
            padding_side="right",
            sequence_length = self.output_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        vid_frames, frames_out = vid_frames.float(), frames_out.float()
        return {
            'inputs': vid_frames,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": "[VIDEO] |", "output": output_text}
        }


class LaSOT(Dataset):
    '''Renders LaSOT tracking tasks as a video sequence'''

    url = "https://drive.google.com/uc?id=1V6ZJH6e4j7N7j8J4zKQ1e2QZ6jFZ2hYn"

    def __init__(self, split, img_type='L', frame_size=64, input_frames=10, sample_frac=1, mask_only=False):
        self.split = split if split != "validation" else "test"
        if split == "val":
            self.split = "test"
        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size
        self.input_frames = input_frames
        self.sample_frac = sample_frac
        self.mask_only = mask_only

        # download from huggingface
        from huggingface_hub import snapshot_download
        if not os.path.exists('datasets/lasot/training_set.txt'):
            snapshot_download(repo_id='l-lt/LaSOT', repo_type='dataset', local_dir='datasets/lasot')

        # Unzip all zip files
        for zip_file in [f for f in os.listdir('datasets/lasot') if f.endswith('.zip')]:
            os.system(f'unzip datasets/lasot/{zip_file} -d datasets/lasot')
            os.system(f'rm datasets/lasot/{zip_file}')

        # Load datasets
        self.data = []
        for line in open(f'datasets/lasot/{self.split}ing_set.txt', 'r'):
            # Check if exists
            if os.path.exists(f'datasets/lasot/{line.strip()}'):
                self.data.append(line.strip())

    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def vocab(self):
        return None

    def __getitem__(self, idx):
        '''Construct video sequence from frames and draw bounding boxes
        Input Sequence: Video
        Output Sequence: Video with bounding boxes'''

        item = self.data[idx % len(self.data)]

        # Load video
        video_dirname = f'datasets/lasot/{item}/img'
        vid_frames = []
        output_frames = []
        bounding_boxes = open(f'datasets/lasot/{item}/groundtruth.txt').readlines()
        good_bounding_boxes = []

        for i, frame_filename in enumerate(sorted(os.listdir(video_dirname))):
            frame = Image.open(os.path.join(video_dirname, frame_filename))
            frame_in = frame.copy().resize((self.frame_size, self.frame_size)).convert(self.img_type)
            if len(vid_frames) > 0:
                vid_frames.append(pil_to_tensor(frame_in).unsqueeze(0))

            x, y, w, h = bounding_boxes[i].split(',')
            if i == 0 or not self.mask_only:
                # Draw bounding boxes for output frames
                draw = ImageDraw.Draw(frame)
                draw.rectangle([int(x), int(y), int(x)+int(w), int(y)+int(h)], outline="red", width=20)

            # Display bounding box on first input frame so model knows which object to track
            if len(vid_frames) == 0:
                vid_frames.append(pil_to_tensor(frame.copy().resize((self.frame_size, self.frame_size)).convert(self.img_type)).unsqueeze(0))

            if self.mask_only:
                # White mask on black background for mask mode
                draw = ImageDraw.Draw(frame)
                draw.rectangle([0, 0, frame.size[0], frame.size[1]], fill="black")
                draw.rectangle([int(x), int(y), int(x)+int(w), int(y)+int(h)], fill="white")


            # Save bounding box for output
            scaled_x = int(int(x) / frame.size[0] * self.frame_size)
            scaled_y = int(int(y) / frame.size[1] * self.frame_size)
            scaled_w = int(int(w) / frame.size[0] * self.frame_size)
            scaled_h = int(int(h) / frame.size[1] * self.frame_size)
            good_bounding_boxes.append((scaled_x, scaled_y, scaled_w, scaled_h))

            frame = frame.resize((self.frame_size, self.frame_size)).convert(self.img_type)
            output_frames.append(pil_to_tensor(frame).unsqueeze(0))

            if len(vid_frames) >= self.input_frames - 1:
                break

        # Subsample to fit in fixed number of frames
        num_vid_frames = self.input_frames - 1
        subsample = len(vid_frames) // num_vid_frames
        subsample = max(subsample, 1)
        vid_frames = vid_frames[::subsample]
        output_frames = output_frames[::subsample]
        vid_frames = vid_frames[:num_vid_frames]
        output_frames = output_frames[:num_vid_frames]
        num_vid_frames = len(vid_frames)
        vid_frames = torch.cat(vid_frames, dim=0)
        output_frames = torch.cat(output_frames, dim=0)

        # Subsample bounding boxes
        good_bounding_boxes = good_bounding_boxes[::subsample]
        good_bounding_boxes = good_bounding_boxes[:num_vid_frames]

        # Add a separator token
        sep = tokens_to_frames(
            ["|"],
            sequence_length = 1,
            img_type = self.img_type,
            frame_size = self.frame_size
        )
        vid_frames = torch.cat([vid_frames, sep], dim=0)

        # Pad and truncate
        if vid_frames.shape[0] < self.input_frames:
            pad_frames = tokens_to_frames(
                [],
                sequence_length = self.input_frames - len(vid_frames),
                img_type = self.img_type,
                frame_size = self.frame_size
            ).squeeze(0)
            vid_frames = torch.cat([pad_frames, vid_frames], dim=0)
            output_frames = torch.cat([pad_frames, output_frames], dim=0)

        vid_frames, output_frames = vid_frames.float(), output_frames.float()
        return {
            'inputs': vid_frames,
            'targets': output_frames,
            'sequence_name': str(idx),
            'config': {
                "input": "[VIDEO] |",
                "output": "[VIDEO]",
                'bounding_boxes': good_bounding_boxes,
            }
        }


class Paraphrase(Dataset):
    '''Renders the MRPC benchmark tasks as a video sequence'''
    def __init__(self, split, img_type='L', frame_size=64, input_frames=10, output_frames=10, sample_frac=1):

        if split == "val":
            split = "validation"

        self.dataset = load_dataset('glue', 'mrpc')[split]
        self.tokenizer = BertPreTokenizer()
        self.input_frames = input_frames
        self.output_frames = output_frames

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size
        self.sample_frac = sample_frac

        # Rename columns
        self.dataset = self.dataset.rename_column('sentence1', 'input')
        self.dataset = self.dataset.rename_column('sentence2', 'output')

    def vocab(self):
        return None

    def __len__(self):
        return int(len(self.dataset) * self.sample_frac)

    def __getitem__(self, idx):
        '''Encode the text as a video sequence'''

        item = self.dataset[idx % len(self.dataset)]
        input_text = item['input'] + " |"
        input_tokens = self.tokenizer.pre_tokenize_str(input_text)
        input_tokens = [token[0] for token in input_tokens]
        frames_in = tokens_to_frames(
            input_tokens,
            sequence_length = self.input_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        output_text = item['output']
        output_tokens = self.tokenizer.pre_tokenize_str(output_text)
        output_tokens = [token[0] for token in output_tokens]
        frames_out = tokens_to_frames(
            output_tokens,
            img_type = self.img_type,
            frame_size = self.frame_size,
            sequence_length = self.output_frames,
            padding_side="right"

        )
        frames_in, frames_out = frames_in.float(), frames_out.float()
        # frames = torch.cat((frames_in, frames_out), 0)
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": input_text, "output": output_text}
        }

class Translation(Dataset):
    '''Renders the MRPC benchmark tasks as a video sequence'''
    def __init__(self, split, source_language="de", target_language="en", img_type='L', frame_size=64, input_frames=10, output_frames=10, sample_frac=1, streaming=False):

        if split == "val":
            split = "validation"

        if streaming and not split == "train":
            print("Disabling streaming for non-training splits")
            streaming = False

        self.dataset = load_dataset('wmt/wmt19', source_language + '-' + target_language)[split]
        self.tokenizer = BertPreTokenizer()
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.source_language = source_language
        self.target_language = target_language
        self.split = split

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size
        self.sample_frac = sample_frac

        # Rename columns
        if streaming:
            self.dataset = iter(self.dataset)

    def vocab(self):
        return None

    def __len__(self):
        return int(len(self.dataset) * self.sample_frac)

    def __getitem__(self, idx):
        '''Encode the text as a video sequence'''

        item = self.dataset[idx % len(self.dataset)]
        input_text = item['translation'][self.source_language] + " |"
        input_tokens = self.tokenizer.pre_tokenize_str(input_text)
        input_tokens = [token[0] for token in input_tokens]
        frames_in = tokens_to_frames(
            input_tokens,
            sequence_length = self.input_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        output_text = item['translation'][self.target_language]
        output_tokens = self.tokenizer.pre_tokenize_str(output_text)
        output_tokens = [token[0] for token in output_tokens]
        frames_out = tokens_to_frames(
            output_tokens,
            img_type = self.img_type,
            frame_size = self.frame_size,
            sequence_length = self.output_frames,
            padding_side="right"

        )
        frames_in, frames_out = frames_in.float(), frames_out.float()
        # frames = torch.cat((frames_in, frames_out), 0)
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": input_text, "output": output_text}
        }

class AudioMNIST(Dataset):
    '''Renders the AudioMNIST dataset as a video sequence'''
    def __init__(self, split, img_type='greyscale', frame_size=64, render_type='spectrogram', sample_frac=1):

        url = "https://github.com/soerenab/AudioMNIST/zipball/master"

        # Load dataset
        if not os.path.exists('datasets/audiomnist'):
            os.makedirs('datasets/audiomnist')
            os.system(f'wget {url} -O datasets/audiomnist.zip')
            os.system(f'unzip datasets/audiomnist.zip -d datasets/audiomnist')
            os.system(f'mv datasets/audiomnist/soerenab-AudioMNIST-*/* datasets/audiomnist')
            os.system(f'rm -r datasets/audiomnist/soerenab-AudioMNIST-*')
            os.system(f'rm datasets/audiomnist.zip')

        base_dir = 'datasets/audiomnist/data'
        self.data = []
        num_speakers = len(os.listdir(base_dir))
        split_point = int(num_speakers * 0.2)
        for i, speaker_dir in enumerate(os.listdir(base_dir)):
            if split == 'train' and i > split_point:
                continue
            if split in ["validation", "val", "test"] and i <= split_point:
                continue
            speaker_dir = os.path.join(base_dir, speaker_dir)
            if os.path.isdir(speaker_dir):
                for filename in os.listdir(speaker_dir):
                    if filename.endswith('.wav'):
                        self.data.append(os.path.join(speaker_dir, filename))

        # Train/test split
        if split == 'train':
            self.data = self.data[:int(len(self.data) * 0.8)]
        if split in ["validation", "val", "test"]:
            self.data = self.data[int(len(self.data) * 0.8):]

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'

        self.frame_size = frame_size
        self.render_type = render_type
        self.split = split
        self.sample_frac = sample_frac

    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def vocab(self):
        return [str(i) for i in range(10)]

    def __getitem__(self, idx):
        '''Encode the audio as a video sequence'''

        item = self.data[idx % len(self.data)]
        target_text = item.split('/')[-1].split('_')[0]

        if self.render_type == 'spectrogram':
            # Render audio as spectrogram
            import torchaudio
            waveform, sample_rate = torchaudio.load(item)
            specgram = torchaudio.transforms.MelSpectrogram()(waveform)
            specgram = torchaudio.transforms.AmplitudeToDB()(specgram)
            specgram = specgram.squeeze(0)
            specgram = specgram - specgram.min()
            specgram = specgram / specgram.max()
            specgram = specgram * 255
            specgram = specgram.int()
            specgram = Image.fromarray(specgram.numpy())
            specgram = specgram.convert(self.img_type)
            specgram = specgram.resize((self.frame_size, self.frame_size))
            audio_image = specgram
        if self.render_type == 'waveform':
            # Render audio as waveform
            import torchaudio
            import matplotlib.pyplot as plt
            waveform, sample_rate = torchaudio.load(item)

            # Plot waveform using matplotlib and convert to PIL image
            ax = plt.gca()
            ax.plot(waveform.squeeze(0).numpy(), color='black')
            # plt.plot(waveform.squeeze(0).numpy(), color='white')
            plt.axis('off')


            import io
            buf = io.BytesIO()
            # fig = plt.gcf()
            # fig.set_size_inches(self.frame_size, self.frame_size)
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            waveform = Image.open(buf)
            waveform = waveform.convert(self.img_type)
            waveform = waveform.resize((self.frame_size, self.frame_size))
            waveform = Image.fromarray(255 - np.array(waveform))
            audio_image = waveform


        # Convert to tensor
        audio_image = pil_to_tensor(audio_image).unsqueeze(0)

        # Add a separator token
        frames_in = tokens_to_frames(
            ["|"],
            sequence_length = 1,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        frames_in = torch.cat([audio_image, frames_in], dim=0)
        frames_in = frames_in.float()

        # Render target text as video sequence
        frames_out = tokens_to_frames(
            [target_text],
            img_type = self.img_type,
            frame_size = self.frame_size,
            padding_side="right",
            sequence_length = 1
        )
        frames_out = frames_out.float()

        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": "[IMG]", "output": target_text}
        }



class FreeSpokenDigit(Dataset):
    '''Renders the free-spoken-digit dataset as a video sequence'''
    def __init__(self, split, img_type='greyscale', frame_size=64, render_type='spectrogram', sample_frac=1):

        url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.zip"

        # Load dataset
        if not os.path.exists('datasets/free-spoken-digit-dataset'):
            os.makedirs('datasets/free-spoken-digit-dataset')
            os.system(f'wget {url} -O datasets/free-spoken-digit-dataset.zip')
            os.system(f'unzip datasets/free-spoken-digit-dataset.zip -d datasets/free-spoken-digit-dataset')
            os.system(f'mv datasets/free-spoken-digit-dataset/free-spoken-digit-dataset-1.0.10/* datasets/free-spoken-digit-dataset')
            os.system(f'rm -r datasets/free-spoken-digit-dataset/free-spoken-digit-dataset-1.0.10')
            os.system(f'rm datasets/free-spoken-digit-dataset.zip')

        base_dir = 'datasets/free-spoken-digit-dataset/recordings'
        self.data = []
        for filename in os.listdir(base_dir):
            if filename.endswith('.wav'):
                self.data.append(os.path.join(base_dir, filename))

        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'


        self.frame_size = frame_size
        self.render_type = render_type
        self.split = split
        self.sample_frac = sample_frac

    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def vocab(self):
        return [str(i) for i in range(10)]

    def __getitem__(self, idx):
        '''Encode the audio as a video sequence'''

        item = self.data[idx % len(self.data)]
        target_text = item.split('/')[-1].split('_')[0]

        if self.render_type == 'spectrogram':
            # Render audio as spectrogram
            import torchaudio
            waveform, sample_rate = torchaudio.load(item)
            specgram = torchaudio.transforms.MelSpectrogram()(waveform)
            specgram = torchaudio.transforms.AmplitudeToDB()(specgram)
            specgram = specgram.squeeze(0)
            specgram = specgram - specgram.min()
            specgram = specgram / specgram.max()
            specgram = specgram * 255
            specgram = specgram.int()
            specgram = Image.fromarray(specgram.numpy())
            specgram = specgram.convert(self.img_type)
            specgram = specgram.resize((self.frame_size, self.frame_size))
            audio_image = specgram
        if self.render_type == 'waveform':
            # Render audio as waveform
            import torchaudio
            import matplotlib.pyplot as plt
            waveform, sample_rate = torchaudio.load(item)

            # Plot waveform using matplotlib and convert to PIL image
            ax = plt.gca()
            ax.plot(waveform.squeeze(0).numpy(), color='black')
            # plt.plot(waveform.squeeze(0).numpy(), color='white')
            plt.axis('off')


            import io
            buf = io.BytesIO()
            # fig = plt.gcf()
            # fig.set_size_inches(self.frame_size, self.frame_size)
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            waveform = Image.open(buf)
            waveform = waveform.convert(self.img_type)
            waveform = waveform.resize((self.frame_size, self.frame_size))
            waveform = Image.fromarray(255 - np.array(waveform))
            audio_image = waveform


        # Convert to tensor
        audio_image = pil_to_tensor(audio_image).unsqueeze(0)

        # Add a separator token
        frames_in = tokens_to_frames(
            ["|"],
            sequence_length = 1,
            img_type = self.img_type,
            frame_size = self.frame_size
        )

        frames_in = torch.cat([audio_image, frames_in], dim=0)
        frames_in = frames_in.float()

        # Render target text as video sequence
        frames_out = tokens_to_frames(
            [target_text],
            img_type = self.img_type,
            frame_size = self.frame_size,
            padding_side="right",
            sequence_length = 1
        )
        frames_out = frames_out.float()

        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": "[IMG]", "output": target_text}
        }


class VideoColorization(Dataset):
    '''Renders video colorization tasks as a video sequence'''
    def __init__(self, split, img_type='rgb', frame_size=64, input_frames=2, output_frames=1, sample_frac=1):
        if img_type == 'greyscale':
            raise ValueError("Image type must be RGB for video colorization tasks")

        if split in ["val", "validation"]:
            split = "test"

        self.tokenizer = BertPreTokenizer()
        self.frame_size = frame_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.sample_frac = sample_frac
        self.split = split
        TinyVIRAT()
        self.data = json.load(open(f'datasets/tinyvirat/tiny_{split}.json'))["tubes"]
        assert input_frames == output_frames + 1

    def __len__(self):
        return int(len(self.data) * self.sample_frac)

    def vocab(self):
        return None

    def __getitem__(self, idx):
        '''Encode the image as a video sequence followed by the label'''

        item = self.data[idx % len(self.data)]

        # Load video
        video_filename = item['path']
        video = cv2.VideoCapture(f'datasets/tinyvirat/videos/{self.split}/{video_filename}')
        input_frames = []
        output_frames = []

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to greyscale and back to RGB
            frame_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_in = cv2.cvtColor(frame_in, cv2.COLOR_GRAY2RGB)

            frame_in = Image.fromarray(frame_in)
            frame_out = Image.fromarray(frame_out)
            frame_in = frame_in.resize((self.frame_size, self.frame_size))
            frame_out = frame_out.resize((self.frame_size, self.frame_size))
            frame_in = pil_to_tensor(frame_in)
            frame_out = pil_to_tensor(frame_out)
            input_frames.append(frame_in)
            output_frames.append(frame_out)

            if len(input_frames) >= self.input_frames - 1:
                break

        input_frames = torch.stack(input_frames, dim=0)
        output_frames = torch.stack(output_frames, dim=0)

        # Add a separator token
        sep = tokens_to_frames(
            ["|"],
            sequence_length = 1,
            img_type = 'RGB',
            frame_size = self.frame_size
        )
        input_frames = torch.cat([input_frames, sep], dim=0)

        return {
            'inputs': input_frames,
            'targets': output_frames,
            'sequence_name': str(idx),
            'config': {"input": "[VIDEO] |", "output": "[VIDEO]"}
        }


class RawText(Dataset):
    '''Renders raw text as a video sequence for use as language model style pretraining'''
    def __init__(self, dataset_name, subset, split, img_type='greyscale', frame_size=64, input_frames=20, sample_frac=1, streaming=True, output_frames=1):
        if img_type == 'greyscale':
            self.img_type = 'L'
        elif img_type == 'rgb':
            self.img_type = 'RGB'
        self.frame_size = frame_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.tokenizer = BertPreTokenizer()
        self.streaming = streaming
        self.sample_frac = sample_frac

        if split == "valid" or split == "val":
            split = "validation"

        self.data = load_dataset(dataset_name, subset, split=split, streaming=streaming)
        self.length = self.data.info.splits[split].num_examples

        # Load text
        if not streaming:
            self.data = self.data.take(self.length * self.sample_frac)
            self.data = list(self.data)
        else:
            self.data = iter(self.data)

    def __len__(self):
        return int(self.length * self.sample_frac)

    def vocab(self):
        return None

    def __getitem__(self, idx):

        if isinstance(self.data, list):
            text = self.data[idx]['text']
        else:
            text = next(self.data)["text"]

        tokens = self.tokenizer.pre_tokenize_str(text)
        tokens = [token[0] for token in tokens]

        total_frames = self.input_frames + self.output_frames

        # Get random subset of tokens of length input_frames
        if len(tokens) - total_frames > 0:
            rand_idx = np.random.randint(0, len(tokens) - total_frames)
            input_tokens = tokens[rand_idx:rand_idx+self.input_frames]
            output_tokens = tokens[rand_idx+self.input_frames:rand_idx+total_frames]
        else:
            input_tokens = tokens[:-self.output_frames]
            output_tokens = tokens[-self.output_frames:]

        frames_in = tokens_to_frames(
            input_tokens,
            sequence_length = self.input_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )
        frames_in = frames_in.float()

        frames_out = tokens_to_frames(
            output_tokens,
            sequence_length = self.output_frames,
            img_type = self.img_type,
            frame_size = self.frame_size
        )
        frames_out = frames_out.float()
        return {
            'inputs': frames_in,
            'targets': frames_out,
            'sequence_name': str(idx),
            'config': {"input": input_tokens, "output": output_tokens}
        }





if __name__ == "__main__":
    ds = ImageToText('cifar10', split='train', img_type='rgb', frame_size=64)
    # ds = CLEVRER(split="train", img_type="rgb", frame_size=64, input_frames=50, question_frames=20)
    #ds = TinyVIRAT(split="train", img_type="rgb", frame_size=64, input_frames=10, output_frames=5)
    #ds = Augmentations_Wrapper(ds, [0])
    # ds = Multitask_Wrapper([
    #     ImageToText('cifar10', split='train', img_type='rgb', frame_size=64),
    #     GLUE('sst2', 'train','greyscale', 20, 64)
    # ])
    #ds = FreeSpokenDigit("validation", "rgb", 64, "waveform", 1)
    #ds = ImageToText("Hamdy20002/COCO_Person", subset=None, split="train", img_type="rgb", output_frames=10)
    #ds = ImageToText("evanarlian/imagenet_1k_resized_256", subset=None, split="train", img_type="rgb", output_frames=10, streaming=True)
    #ds = ImageToText("google-research-datasets/conceptual_captions", subset="unlabeled", split="train", img_type="rgb", output_frames=20, streaming=True)
    #ds = VideoColorization(split="train", img_type="rgb", frame_size=64, input_frames=50, output_frames=49)
    #ds = Cache_Wrapper(ds)
    #ds = Translation("train", "de", "en", "rgb", 64, 20, 20)

    #ds = LaSOT(split="train", img_type="rgb", frame_size=64, input_frames=10)
    # ds = RawText("allenai/c4", "en", "train", "greyscale", 64, 20, 50000)
    #ds = RawText("allenai/c4", subset="en", split="train", img_type="greyscale", frame_size=64, input_frames=20, streaming=True, sample_frac=0.01)
    #ds = CLEVRER("train", "rgb", 64, 50)
    ds = AudioMNIST("train", "rgb", 64, "spectrogram", 1)

    # Save example back to disk
    if not os.path.exists('./test'):
        os.makedirs('./test')

    idx = 1
    example = ds[idx]
    breakpoint()

    for i, frame in enumerate(example['inputs']):
        if torch.max(frame) > 1:
            frame = frame / 255

        to_pil_image(frame).save(f'./test/{i:03d}.png')

    if example['targets'] is not None:
        for j, frame in enumerate(example['targets']):
            if torch.max(frame) > 1:
                frame = frame / 255
            to_pil_image(frame).save(f'./test/{i+j+1:03d}.png')
