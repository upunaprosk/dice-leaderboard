import datasets
import os
from datasets import load_dataset, concatenate_datasets
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset

DEFAULT_DATA_DIR = "../data/pretrain_datasets/roberta"
DEFAULT_HUB_URL = "gsgoncalves/roberta_pretrain"
DEFAULT_ROBERTA_MODEL = 'roberta-base'

class RobertaPretrainDatasetHub(Dataset):
	def __init__(self, hub_url=DEFAULT_HUB_URL, roberta_model=DEFAULT_ROBERTA_MODEL):
		self.dataset = load_dataset(hub_url)
		self.dataset = self.dataset['train']#.shuffle()
		self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_model)

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


class RobertaPretrainDataset(Dataset):
	def __init__(self, data_dir=DEFAULT_DATA_DIR):
		if data_dir and os.path.exists(data_dir) and os.path.isdir(data_dir):
			self.dataset = datasets.load_from_disk(data_dir)
			self.data_dir = data_dir
		else:
			bookcorpus = load_dataset("bookcorpus", split="train")
			openweb = load_dataset("openwebtext", split="train")
			# cc_stories = load_dataset("spacemanidol/cc-stories", split="train")
			cc_news = load_dataset("cc_news", split="train")
			# only keep the 'text' column
			cc_news = cc_news.remove_columns([col for col in cc_news.column_names if col != "text"])

			assert bookcorpus.features.type == openweb.features.type
			assert bookcorpus.features.type == cc_news.features.type
			#assert bookcorpus.features.type == cc_stories.features.type
			self.dataset = concatenate_datasets([bookcorpus, openweb, cc_news])
			#self.dataset = concatenate_datasets([bookcorpus, openweb, cc_news, cc_stories])
			self.dataset = self.dataset.shuffle()
			os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
			self.dataset.save_to_disk(DEFAULT_DATA_DIR)
		self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

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
