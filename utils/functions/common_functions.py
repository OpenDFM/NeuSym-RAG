#coding=utf8
import time, uuid, re, tiktoken
from tiktoken import Encoding
from typing import List, Dict, Union, Optional, Any, Iterable
from agents.models import get_llm_single_instance


ENCODING_MODELS = dict()

def truncate_tokens(text: str, max_tokens: int = 30, encoding_model: str = 'cl100k_base') -> str:
    """ Given a text string, truncate it to max_tokens * 1000 using encoding_model tokenizer
    """
    if encoding_model not in ENCODING_MODELS:
        encoding: Encoding = tiktoken.get_encoding(encoding_model)
        ENCODING_MODELS[encoding_model] = encoding
    encoding: Encoding = ENCODING_MODELS[encoding_model]
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens * 1000:
        tokens = tokens[:max_tokens * 1000]
        text = encoding.decode(tokens)
    return text


def call_llm_with_message(
        messages: List[Dict[str, Any]], 
        model: str = 'gpt-4o-mini', 
        top_p: float = 0.95, 
        temperature: float = 0.7
    ) -> str:
    """ Call LLM to generate the response directly using the message list.
    """
    model_client = get_llm_single_instance(model)
    response = model_client.get_response(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p
    )
    time.sleep(1)
    return response


def convert_to_message(
        template: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
    """ Convert the template to the message list. The `template` merely supports the following format:
    {{system_message}}

    {{user_message}}
    Note that, the system and user messages should be separated by two consecutive newlines. And the first block is the system message, the other blocks are the user message. There is no assistant message or interaction history.
    
    If you need to add an image, you can set `image` in kwargs to the image message dict. See `get_image_message` in `utils.functions.image_functions` for the image message format.
    """
    system_msg = template.split('\n\n')[0].strip()
    user_msg = '\n\n'.join(template.split('\n\n')[1:]).strip()
    messages = [
        {
            "role": 'system',
            "content": system_msg
        }
    ]
    if user_msg:
        messages.append(
            {
                "role": 'user',
                "content": user_msg
            }
        )
    if kwargs.get("image", None):
        image_message = kwargs["image"]
        if type(image_message) is list:
            messages.extend(image_message)
        else:
            messages.append(image_message)
    return messages


def call_llm(
        template: str, 
        model: str = 'gpt-4o-mini', 
        top_p: float = 0.95, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
    """ Automatically construct the message list from template and call LLM to generate the response. 
    See `convert_to_message` for the template format.
    """
    return call_llm_with_message(messages=convert_to_message(template, **kwargs), model=model, top_p=top_p, temperature=temperature)


def call_llm_with_pattern(
        template: str,
        pattern: str, 
        model: str = 'gpt-4o-mini',
        top_p: float = 0.95, 
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
    """ Automatically construct the message list from template, call LLM to generate the response, and parse the response with givern pattern.
    """
    response = call_llm(template=template, model=model, top_p=top_p, temperature=temperature, **kwargs)
    matched = re.findall(pattern, response, re.DOTALL)
    if len(matched) == 0:
        return None
    return [s.strip() for s in matched[-1]]


def get_uuid(name: Optional[str] = None, uuid_type: str = 'uuid5', uuid_namespace: str = 'dns') -> str:
    """ Generate a UUID string given the input name.
    @args:
        name: str or Iterable[str], the input name to generate the UUID.
        uuid_type: str, chosen from uuid3, uuid4, uuid5, default to uuid5.
            - uuid4: Generate a random UUID, no input args.
            - uuid3: Generate a UUID based on the MD5 hash of the input name. Always return the same UUID for the same input (name, uuid_namespace).
            - uuid5: Generate a UUID based on the SHA-1 hash of the input name. Always return the same UUID for the same input (name, uuid_namespace).
        uuid_namespace: str, chosen from dns, url, oid, x500, default to dns.
    @return:
        uid: str, return the string format of the uuid.
    """
    namespaces = {'dns': uuid.NAMESPACE_DNS, 'url': uuid.NAMESPACE_URL, 'oid': uuid.NAMESPACE_OID, 'x500': uuid.NAMESPACE_X500}
    namespace = namespaces[uuid_namespace.lower()]
    if uuid_type == 'uuid3' or uuid_type == 'uuid5':
        if name is None:
            raise ValueError('The input name should not be None for uuid3 or uuid5.')

        uid = uuid.uuid5(namespace, name) if uuid_type == 'uuid5' else uuid.uuid3(namespace, name)
    else:
        uid = uuid.uuid4()
    return str(uid)


def is_valid_uuid(string: str, version: Optional[int] = None) -> bool:
    """ Determine whether a string is a valid UUID. The version parameter is optional.
    """
    try:
        uuid_obj = uuid.UUID(string)
        if version is not None:
            return int(uuid_obj.version) == int(version)
        return True
    except Exception as e:
        return False
