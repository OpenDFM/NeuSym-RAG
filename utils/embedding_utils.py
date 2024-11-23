#coding=utf8
import os, tempfile
import numpy as np
from typing import List, Dict, Union, Any
from towhee import DataCollection, ops, pipe
from towhee.runtime.runtime_pipeline import RuntimePipeline
from milvus_model.base import BaseEmbeddingFunction
from utils.vectorstore_utils import detect_embedding_model_path
from PIL import Image
from pdf2image import convert_from_path


TEMP_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')


def get_clip_image_embedding_pipeline(embed_model: str = 'clip-vit-base-patch16') -> RuntimePipeline:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    embed_model = detect_embedding_model_path(embed_model)
    return pipe.input('path').map('path', 'image', ops.image_decode.cv2('rgb')).map('image', 'vector', ops.image_text_embedding.clip(model_name=embed_model, modality='image')).map('vector', 'vector', lambda x: x / np.linalg.norm(x)).output('vector')


def get_clip_text_embedding_pipeline(embed_model: str = 'clip-vit-base-patch16') -> RuntimePipeline:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    embed_model = detect_embedding_model_path(embed_model)
    return pipe.input('text').map('text', 'vector', ops.image_text_embedding.clip(model_name=embed_model, modality='text')).map('vector', 'vector', lambda x: x / np.linalg.norm(x)).output('vector')


class ClipEmbeddingFunction(BaseEmbeddingFunction):

    def __init__(self, model_name: str = 'clip-vit-base-patch16'):
        self.model_name = os.path.basename(model_name.rstrip(os.sep))
        self.image_embedding_pipeline = get_clip_image_embedding_pipeline(self.model_name)
        self.text_embedding_pipeline = get_clip_text_embedding_pipeline(self.model_name)


    def __call__(self, texts_or_images: List[Union[str, Dict[str, Any]]]):
        """
        """
        if len(texts_or_images) == 0:
            raise ValueError('[Error]: no input to encode!')
        if isinstance(texts_or_images[0], str): # text inputs
            embeddings = self.encode_queries(texts_or_images)
        else:
            embeddings = self.encode_documents(texts_or_images)
        return embeddings


    def encode_queries(self, texts: List[str]) -> List[np.ndarray]:
        return [DataCollection(self.text_embedding_pipeline(query))[0]['vector'] for query in texts]


    def encode_documents(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """ Single wrapper.
        """
        return self.encode_images(documents)


    def encode_images(self, images: List[Dict[str, Any]]) -> List[np.ndarray]:
        """ Encode images, each image object is repsented as a dict with the following keys:
         {
            "path": str, required, the path to the image or PDF file,
            "page": int, optional, used for PDF file,
            "bbox": List[int], optional, used to crop the image or PDF page
         }
        Note that, `page` starts from 1, and `bbox` is a tuple of length 4 representing (x0, y0, width, height).
        """
        embeddings = []
        for image_obj in images:
            image_path = image_obj["path"]
            if image_path.endswith('.pdf'): # PDF path, must specify the page number
                page_number = int(image_obj["page"])
                image = convert_from_path(image_path)[page_number - 1]
            else: image = Image.open(image_path)

            if len(image_obj.get("bbox", [])) == 4:
                bbox = list(image_obj["bbox"])
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                image = image.crop(bbox) # (x0, y0, x1, y1)

            with tempfile.NamedTemporaryFile(suffix='.png', dir=TEMP_CACHE_DIR) as image_file:
                image.save(image_file.name, 'PNG')
                embeddings.append(DataCollection(self.image_embedding_pipeline(image_file.name))[0]['vector'])

        return embeddings