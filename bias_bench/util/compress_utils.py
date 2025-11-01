import random

import torch
import os
import time
import copy

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from transformers import BertModel, BertForMaskedLM
from bias_bench.dataset.bert_pretrain_dataset import BertPretrainDatasetHub
from bias_bench.dataset.roberta_pretrain_dataset import RobertaPretrainDatasetHub

DYNAMIC = 'dynamic'
STATIC = 'static'
QAT = 'qat'
PTQ = 'ptq'

def quantize_static(model, precision=None):
	if isinstance(model, BertModel) or isinstance(model, BertForMaskedLM):
		dataset = BertPretrainDatasetHub()
	else:
		dataset = RobertaPretrainDatasetHub()
	dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	# Start quantization calibration
	m = copy.deepcopy(model)
	m.eval()
	#m.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
	m.qconfig = torch.ao.quantization.default_qconfig
	print(m.qconfig)
	m_prep = torch.ao.quantization.prepare(m, inplace=False)

	# The following is a hack because Embedding layers do not support int quantization atm.
	for _, mod in m_prep.named_modules():
		if isinstance(mod, torch.nn.Embedding):
			mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

	#qconfig_dict = {"": torch.quantization.get_default_qconfig(torch.backends.quantized.engine)}
	# prepare the model for quantization
	#first_sample = next(iter(dataloader))
	#TODO this is not working
	#example_inputs = (first_sample['input_ids'], first_sample['attention_mask'], first_sample['token_type_ids'])
	# m_prepared = prepare_fx(m, qconfig_dict, example_inputs)

	# Calibrate the model
	with torch.inference_mode():
		total_steps = 2# 625 # 625*16 = 10k samples
		curr_step = 0
		for i, batch in tqdm(enumerate(dataloader), desc='Static Quantization', total=total_steps):
			batch = {k:v.squeeze(1).to(device) for k, v in batch.items()}
			# m_prepared(**batch)
			#m_prepared(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
			m_prep(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
			curr_step += 1
			if curr_step >= total_steps:
				break

	# quantize
	# static_quantized_model = convert_fx(m_prepared).to(device)
	m_quantized = torch.ao.quantization.convert(m_prep, inplace=False)
	#del m
	#del m_prepared
	#return static_quantized_model
	return m_quantized


def time_model_evaluation(model, configs, tokenizer):
	eval_start_time = time.time()
	result = evaluate(configs, model, tokenizer, prefix="")
	eval_end_time = time.time()
	eval_duration_time = eval_end_time - eval_start_time
	print(result)
	print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


def print_size_of_model(model):
	torch.save(model.state_dict(), "temp.p")
	mb = os.path.getsize("temp.p") / 1e6
	print('Size (MB):', mb)
	os.remove('temp.p')
	return mb


# Avg model sparsity
def calculate_sparsity(model):
	sparsity_per_pruned_layer = [(name, 100.0 * float(torch.sum(buffer == 0)) / float(buffer.nelement()))
	                             for name, buffer in model.named_buffers()]
	avg_pruned_sparsity = sum([x[1] for x in sparsity_per_pruned_layer]) / float(len(sparsity_per_pruned_layer))
	pruned_layer_names = {x[0].split('.')[0] for x in sparsity_per_pruned_layer}
	non_pruned_layers = [parameter.data for parameter in model.named_parameters() if
	                     parameter[0].split('.')[0] not in pruned_layer_names and 'bias' not in parameter[0]]
	overall_sparsity = sum([x[1] for x in sparsity_per_pruned_layer]) / float(
		len(sparsity_per_pruned_layer) + len(non_pruned_layers))

	print(f"Sparsity per pruned layer: {sparsity_per_pruned_layer}")
	print(f"Avg Sparsity per pruned layer: {avg_pruned_sparsity} %")
	print(f"Overall Sparsity per pruned layer: {overall_sparsity} %")

	return sparsity_per_pruned_layer, avg_pruned_sparsity, overall_sparsity


def parse_precision(precision: str):
	if precision == 'fp16':
		return torch.float16
	#elif precision == 'fp32':
#		return torch.float
	elif precision == 'int8':
		return torch.qint8
	else:
		raise ValueError(f"Precision {precision} not supported. Try fp16, or int8.")