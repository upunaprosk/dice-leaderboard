import copy
from functools import partial
from typing import Union

import torch
import transformers

from bias_bench.debias.self_debias.modeling import GPT2Wrapper, OPTWrapper, LLamaWrapper
from bias_bench.debias.self_debias.modeling import MaskedLMWrapper
from bias_bench.util.compress_utils import DYNAMIC, QAT, PTQ, parse_precision, STATIC, quantize_static

n_gpus = torch.cuda.device_count()

class QuantizedBertModel:
    def __new__(self, model_name_or_path, precision,#: Union[torch.qint8, torch.qint32, torch.float16],
                quant_type: str = DYNAMIC):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        precision = parse_precision(precision)
        if quant_type == DYNAMIC:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        else:
            raise NotImplementedError
        self.precision = precision
        self.quant_type = quant_type
        return model

class QuantizedBertModelMaskedLM:
    def __new__(self, model_name_or_path, precision,#: Union[torch.qint8, torch.float16],
                quant_type: str = DYNAMIC):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        precision = parse_precision(precision)
        if quant_type == DYNAMIC:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        elif quant_type == STATIC:
            model = quantize_static(model, precision)
        else:
            raise NotImplementedError
        self.precision = precision
        self.quant_type = quant_type
        return model

class QuantizedRobertaModel:
    def __new__(self, model_name_or_path, precision,#: Union[torch.qint8, torch.qint32, torch.float16],
                quant_type: str = DYNAMIC):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        precision = parse_precision(precision)
        if quant_type == DYNAMIC:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        else:
            raise NotImplementedError
        self.precision = precision
        self.quant_type = quant_type
        return model

class QuantizedRobertaModelMaskedLM:
    def __new__(self, model_name_or_path, precision,#: Union[torch.qint8, torch.float16],
                quant_type: str = DYNAMIC):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        precision = parse_precision(precision)
        if quant_type == DYNAMIC:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        elif quant_type == STATIC:
            model = quantize_static(model, precision)
        else:
            raise NotImplementedError
        self.precision = precision
        self.quant_type = quant_type
        return model

class IBertModel:
    def __new__(self, model_name_or_path):
        return transformers.IBertModel.from_pretrained(model_name_or_path)

class IBertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.IBertForMaskedLM.from_pretrained(model_name_or_path)

class DistilBertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)

class DistilRobertModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)

class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)

class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)



class AutoModelForCausalLM:
    def __new__(self, model_name_or_path):
        return transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, output_hidden_states=True)


class AlbertModel:
    def __new__(self, model_name_or_path):
        return transformers.AlbertModel.from_pretrained(model_name_or_path)


class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)


class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


class DistilBertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


class AlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)


class DistilRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)


class RobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)


class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)


class GPTNeoXModel:
    def __new__(self, model_name_or_path, revision, cache_dir=None):
        if cache_dir:
            model = transformers.GPTNeoXModel.from_pretrained(model_name_or_path, revision=revision, cache_dir=cache_dir)
        else:
            model = transformers.GPTNeoXModel.from_pretrained(model_name_or_path, revision=revision)
        self.revision = revision
        self.cache_dir = cache_dir
        return model

class GPTNeoXForCausalLM:
    def __new__(self, model_name_or_path, revision, cache_dir=None):
        if cache_dir:
            model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name_or_path, revision=revision,
                cache_dir=cache_dir)
        else:
            model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name_or_path, revision=revision)
        self.revision = revision
        self.cache_dir = cache_dir
        return model

class QuantizedGPTNeoXModel:
    def __new__(self, model_name_or_path, revision, cache_dir=None, precision='int8',#: Union[torch.qint8, torch.qint32, torch.float16],
                quant_type: str = DYNAMIC):
        if cache_dir:
            model = transformers.GPTNeoXModel.from_pretrained(model_name_or_path, revision=revision,
                                                              cache_dir=cache_dir)
        else:
            model = transformers.GPTNeoXModel.from_pretrained(model_name_or_path, revision=revision)
        precision = parse_precision(precision)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        self.quant_type = quant_type
        self.precision = precision
        self.revision = revision
        self.cache_dir = cache_dir
        return model

class QuantizedGPTNeoXForCausalLM:
    def __new__(self, model_name_or_path, revision, cache_dir=None, precision='int8',#: Union[torch.qint8, torch.float16],
                quant_type: str = DYNAMIC):
        if cache_dir:
            model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name_or_path, revision=revision, cache_dir=cache_dir)
        else:
            model = transformers.GPTNeoXForCausalLM.from_pretrained(model_name_or_path, revision=revision)
        precision = parse_precision(precision)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=precision)
        self.quant_type = quant_type
        self.precision = precision
        self.revision = revision
        self.cache_dir = cache_dir
        return model

class _SentenceDebiasModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, input_, output, bias_direction):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            bias_direction = bias_direction.to(x.device)

            # Debias the representation.
            for t in range(x.size(1)):
                x[:, t] = x[:, t] - torch.ger(
                    torch.matmul(x[:, t], bias_direction), bias_direction
                ) / bias_direction.dot(bias_direction)

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, bias_direction=bias_direction)


class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output
            opt_or_llama=False
            if "last_hidden_state" not in output.keys():
                hidden_states = output.hidden_states
                x = hidden_states[-1]
                opt_or_llama=True
            else:
                x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            if opt_or_llama:
                output.hidden_states[-1] = x
            else:
                # Update the output.
                output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)


class SentenceDebiasBertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2Model(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class SentenceDebiasBertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class INLPAlbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class INLPRobertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class INLPOPTLMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.OPTForCausalLM.from_pretrained(model_name_or_path)
        model.model.decoder.register_forward_hook(self.func)
        return model

class INLPLlamaLMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)
        model.model.register_forward_hook(self.func)
        return model




class CDABertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class CDAAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class CDARobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class CDAGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class CDABertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDARobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class DropoutBertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class DropoutBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class BertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SentenceDebiasBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2ForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2ForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDARobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutBertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutRobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SelfDebiasBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = GPT2Wrapper(model_name_or_path, use_cuda=False)
        return model


class SelfDebiasOPTLMHeadModel:
    def __new__(self, model_name_or_path):
        model = OPTWrapper(model_name_or_path, use_cuda=False)
        return model


class SelfDebiasLLAMALMHeadModel:
    def __new__(self, model_name_or_path):
        model = LLamaWrapper(model_name_or_path, use_cuda=False)
        return model