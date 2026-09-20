import datasets
import os
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = "../data/pretrain_datasets/bert"
DEFAULT_HUB_URL = "gsgoncalves/bert_pretrain"
DEFAULT_BERT_MODEL = 'bert-base-uncased'

class BertPretrainDatasetHub(Dataset):
	def __init__(self, hub_url=DEFAULT_HUB_URL, bert_model=DEFAULT_BERT_MODEL):
		self.dataset = load_dataset(hub_url)
		self.dataset = self.dataset['train']#.shuffle()
		self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		text = self.dataset[idx]['text']
		#tok = self.tokenizer(text, return_special_tokens_mask = True, truncation = True, padding='max_length',
		#               max_length = self.tokenizer.model_max_length, return_tensors='pt')
		#tok = self.tokenizer.encode_plus(text, return_special_tokens_mask=True, truncation=True,
		#                                 padding='max_length', max_length = self.tokenizer.model_max_length, return_tensors="pt")
		tok = self.tokenizer(text, return_token_type_ids=True, return_special_tokens_mask=True,
		                     truncation = True, max_length=self.tokenizer.model_max_length,
		                     padding='max_length', return_tensors="pt")
		return tok

class BertPretrainDataset(Dataset):
	def __init__(self, data_dir=None):
		if data_dir and os.path.exists(data_dir) and os.path.isdir(data_dir):
			self.dataset = datasets.load_from_disk(data_dir)
			self.data_dir = data_dir
		else:
			bookcorpus = load_dataset("bookcorpus", split="train")
			wiki = load_dataset("wikipedia", "20220301.en", split="train")
			# only keep the 'text' column
			wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

			assert bookcorpus.features.type == wiki.features.type
			self.dataset = concatenate_datasets([bookcorpus, wiki])
			self.dataset = self.dataset.shuffle()
			os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
			self.dataset.save_to_disk(DEFAULT_DATA_DIR)
		self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		text = self.dataset[idx]['text']
		#tok = self.tokenizer(text, return_special_tokens_mask = True, truncation = True, padding='max_length',
		#               max_length = self.tokenizer.model_max_length, return_tensors='pt')
		#tok = self.tokenizer.encode_plus(text, return_special_tokens_mask=True, truncation=True,
		#                                 padding='max_length', max_length = self.tokenizer.model_max_length, return_tensors="pt")
		tok = self.tokenizer(text, return_special_tokens_mask=True, truncation = True, max_length=self.tokenizer.model_max_length, padding='max_length')
		return tok
