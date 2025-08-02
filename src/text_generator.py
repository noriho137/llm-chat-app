import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Prompt related definitions
PROMPT = '質問にできるだけ正確に答えてください。\n\n## 質問:\n{question}'
RAG_PROMPT = '参考情報を元にして質問にできるだけ正確に答えてください。\n\n## 参考情報:\n{context}\n\n## 質問:\n{question}'


def make_context(results):
    """
    Convert the results of a vector search into a bulleted format for embedding as reference information in a prompt.
    """
    logger.debug('start')

    context = None

    docs = results['documents'][0]
    if docs:
        context = [doc for doc in docs]
        context = '\n* '.join(context)
        context = '* ' + context

    logger.debug('end')
    return context


class TextGenerator:
    def __init__(self, model_name_or_path, message_config_path=None, quantization_method=None):
        """
        Setup LLM.
        """
        logger.debug('start')
        logger.info(f'model_name_or_path={model_name_or_path}, '
                    f'message_config_path={message_config_path}, '
                    f'quantization_method={quantization_method}')

        # Load pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.chat_template = None
        self.add_generation_prompt = None
        self.messages = None
        self.generate_args = {}

        # Load message config
        if message_config_path:
            with open(message_config_path, 'r') as f:
                message_config = json.load(f)

            self.chat_template = message_config.get('chat_template')
            self.add_generation_prompt = message_config.get('add_generation_prompt')
            self.messages = message_config.get('messages')
            self.generate_args = message_config.get('generate_args')

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

    def make_messages(self, query, context=None):
        """
        Make messages
        """
        messages = self.messages.copy()

        if context is None:
            # Without reference information
            _prompt = PROMPT.format(question=query)
        else:
            # With reference information (RAG)
            _prompt = RAG_PROMPT.format(context=context, question=query)

        for message in messages:
            if message['role'] == 'user':
                message['content'] = _prompt

        return messages

    def make_prompt(self, query, context=None):
        """
        Make a prompt.
        """
        logger.debug('start')

        messages = self.make_messages(query, context)
        logger.debug(f'messages={messages}')

        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt
        )
        logger.debug(prompt)

        logger.debug('end')
        return prompt

    def generate_answer(self, prompt):
        """
        Input a prompt to LLM to generate an answer text.
        """
        logger.debug('start')

        self.model.eval()

        with torch.no_grad():
            token_ids = self.tokenizer.encode(prompt,
                                              add_special_tokens=False,
                                              return_tensors='pt')

            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                **self.generate_args
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
