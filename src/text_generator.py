import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, GPTQConfig

logger = logging.getLogger(__name__)

# Prompt related definitions
B_INST = '[INST]'
E_INST = '[/INST]'
B_SYS = '<<SYS>>\n'
E_SYS = '\n<</SYS>>\n\n'
DEFAULT_SYSTEM_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。質問にできるだけ正確に答えてください。'
DEFAULT_SYSTEM_RAG_PROMPT = 'あなたは誠実で優秀な日本人のアシスタントです。参考情報を元にして質問にできるだけ正確に答えてください。'
PROMPT = '## 質問:\n{question}'
RAG_PROMPT = '## 参考情報:\n{context}\n\n## 質問:\n{question}'

# Maximum length of newly generated tokens
MAX_NEW_TOKENS = 256


def make_context(results):
    """
    Convert the results of a vector search into a bulleted format for embedding as reference information in a prompt.
    """
    logger.debug('start')
    context = [doc for doc in results['documents'][0]]
    context = '\n* '.join(context)
    context = '* ' + context
    logger.debug('end')
    return context


class TextGenerator:
    def __init__(self, model_name_or_path, quantization_method=None):
        """
        Setup LLM.
        """
        logger.debug('start')
        logger.info(f'model_name_or_path={model_name_or_path}, '
                    f'quantization_method={quantization_method}')

        # Load pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Quantization settings
        if quantization_method == 'bitsandbytes':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4',
                                                     bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            quantization_config = None

        # Load pretrained model
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype='auto',
                                                          quantization_config=quantization_config)

        logger.debug('end')
        return

    def make_prompt(self, query, context=None):
        """
        Make a prompt.
        """
        logger.debug('start')

        if context is None:
            # Without reference information
            _system_prompt = DEFAULT_SYSTEM_PROMPT
            _prompt = PROMPT.format(question=query)
        else:
            # With reference information (RAG)
            _system_prompt = DEFAULT_SYSTEM_RAG_PROMPT
            _prompt = RAG_PROMPT.format(context=context, question=query)

        prompt = '{bos_token}{b_inst} {system}{prompt} {e_inst} '.format(
            bos_token=self.tokenizer.bos_token,
            b_inst=B_INST,
            system=f'{B_SYS}{_system_prompt}{E_SYS}',
            prompt=_prompt,
            e_inst=E_INST,
        )
        logger.debug(prompt)

        logger.debug('end')
        return prompt

    def generate_answer(self, prompt):
        """
        Input a prompt to LLM to generate an answer text.
        """
        logger.debug('start')

        with torch.no_grad():
            token_ids = self.tokenizer.encode(prompt,
                                              add_special_tokens=False,
                                              return_tensors='pt')

        output_ids = self.model.generate(
            token_ids.to(self.model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):],
                                       skip_special_tokens=True)

        logger.debug('end')
        return output

    def run(self, query, context):
        """
        Generate an answer text to a query.
        """
        logger.debug('start')

        # Make prompt.
        prompt = self.make_prompt(query=query, context=context)

        # Generate answer text.
        answer = self.generate_answer(prompt=prompt)

        logger.debug('end')
        return answer
