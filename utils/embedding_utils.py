#coding=utf8
import os, tempfile, time
import numpy as np
from typing import List, Dict, Union, Any
from towhee import ops, pipe
from towhee.runtime.runtime_pipeline import RuntimePipeline
from milvus_model.base import BaseEmbeddingFunction
from utils.config import CACHE_DIR, TMP_DIR
from utils.vectorstore_utils import detect_embedding_model_path
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from collections import defaultdict


PDF_TO_IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, 'pdf_to_images')
os.makedirs(PDF_TO_IMAGE_CACHE_DIR, exist_ok=True)


def get_clip_image_embedding_pipeline(embed_model: str = 'clip-vit-base-patch32', device = 'cpu') -> RuntimePipeline:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    embed_model = detect_embedding_model_path(embed_model)
    return pipe.input('path').map('path', 'image', ops.image_decode.cv2('rgb')).map('image', 'vector', ops.image_text_embedding.clip(model_name=embed_model, modality='image', device=device)).map('vector', 'vector', lambda x: x / np.linalg.norm(x)).output('vector')


def get_clip_text_embedding_pipeline(embed_model: str = 'clip-vit-base-patch32', device = 'cpu') -> RuntimePipeline:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    embed_model = detect_embedding_model_path(embed_model)
    return pipe.input('text').map('text', 'vector', ops.image_text_embedding.clip(model_name=embed_model, modality='text', device=device)).map('vector', 'vector', lambda x: x / np.linalg.norm(x)).output('vector')


class ClipEmbeddingFunction(BaseEmbeddingFunction):

    def __init__(self, model_name: str = 'clip-vit-base-patch32', image_batch_size: int = 256, device = 'cpu'):
        self.model_name = os.path.basename(model_name.rstrip(os.sep))
        self.image_batch_size = image_batch_size
        self.image_embedding_pipeline = get_clip_image_embedding_pipeline(self.model_name, device)
        self.text_embedding_pipeline = get_clip_text_embedding_pipeline(self.model_name, device)
        self.pdf_to_images = defaultdict(list)


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
        return [dq.get()[0] for dq in self.text_embedding_pipeline.batch(texts)]


    def encode_query(self, text: str) -> np.ndarray:
        return self.text_embedding_pipeline(text).get()[0]


    def encode_documents(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """ Single wrapper.
        """
        return self.encode_images(documents)


    def cache_pdf_to_images(self, pdf_ids: List[str], pdf_paths: List[str]) -> List[str]:
        """ Cache PDF files to images."""
        # start_time = time.time()
        for pdf_id, pdf_path in zip(pdf_ids, pdf_paths):
            if pdf_path.endswith('.pdf') and pdf_path not in self.pdf_to_images:
                with open(pdf_path, 'rb') as fin:
                    pdf_reader = PdfReader(fin)
                    width_height = [(p.mediabox.width, p.mediabox.height) for p in pdf_reader.pages]
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(pdf_path, output_folder=temp_dir)
                    for i, image in enumerate(images):
                        image = image.convert('RGB')
                        width_ratio, height_ratio = image.width / width_height[i][0], image.height / width_height[i][1]
                        self.pdf_to_images[pdf_path].append((image, width_ratio, height_ratio))
                        # image_path = os.path.join(PDF_TO_IMAGE_CACHE_DIR, f"{pdf_id}_page_{i}.png")
                        # image.save(image_path, 'PNG')
                        # self.pdf_to_images[pdf_path].append((image_path, width_ratio, height_ratio))
        # print(f'Caching {len(pdf_ids)} PDF images costs {time.time() - start_time}s')
        return self.pdf_to_images


    def clear_cache(self):
        # safer way
        # for pdf_path in self.pdf_to_images:
            # for filepath, _, _ in self.pdf_to_images[pdf_path]:
                # if os.path.exists(filepath): os.remove(filepath)
        self.pdf_to_images = defaultdict(list)
        return


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
        for i in range(0, len(images), self.image_batch_size):
            temp_image_files = []
            for image_obj in images[i:i + self.image_batch_size]:
                image_path = image_obj["path"]
                if image_path.endswith('.pdf'): # PDF path, must specify the page number
                    page_number = int(image_obj["page"])
                    image, width_ratio, height_ratio = self.pdf_to_images[image_path][page_number - 1]
                    # image_path, width_ratio, height_ratio = self.pdf_to_images[image_path][page_number - 1]
                else:
                    width_ratio = height_ratio = 1
                    with Image.open(image_path, 'r') as image:
                        image = image.convert('RGB')

                # with Image.open(image_path, 'r') as image:
                if True:
                    if len(image_obj.get("bbox", [])) == 4:
                        bbox = list(image_obj["bbox"])
                        bbox[2] = (bbox[0] + bbox[2]) * width_ratio
                        bbox[3] = (bbox[1] + bbox[3]) * height_ratio
                        bbox[0] *= width_ratio
                        bbox[1] *= height_ratio
                        cropped_image = image.crop(bbox) # (x0, y0, x1, y1)
                    else: cropped_image = image
                    temp_image_files.append(
                        tempfile.NamedTemporaryFile(suffix='.png', dir=TMP_DIR)
                    )
                    cropped_image.save(temp_image_files[-1].name, 'PNG')

            # image batch encoding
            vectors = self.image_embedding_pipeline.batch([t.name for t in temp_image_files])
            embeddings.extend([v.get()[0] for v in vectors])
            for t in temp_image_files: t.close() # close the temp files
        return embeddings
