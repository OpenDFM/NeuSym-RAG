#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import base64
import copy
import gymnasium as gym
import os
import tempfile
from typing import Optional, List, Tuple, Dict, Union, Any
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path


@dataclass
class ViewImage(Action):

    paper_id: str = field(default='', repr=True) # concrete paper id, required
    page_number: int = field(default=1, repr=True) # concrete page number, required
    bounding_box: List[float] = field(default_factory=list, repr=True) # concrete bounding box, optional

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the image.
        """
        try:
            self.paper_id = str(self.paper_id)
        except:
            return Observation('[Error]: paper id must be a string.')
        try:
            self.page_number = int(self.page_number)
            assert self.page_number > 0
        except:
            return Observation('[Error]: page number must be a positive integer.')
        if self.bounding_box is None or self.bounding_box == '':
            self.bounding_box = []
        try:
            assert isinstance(self.bounding_box, list)
            assert len(self.bounding_box) == 0 or len(self.bounding_box) == 4
            for i in range(len(self.bounding_box)):
                self.bounding_box[i] = float(self.bounding_box[i])
        except:
            return Observation('[Error]: bounding box must be a list of 0 or 4 floats.')

        if env.dataset == 'airqa':
            pdf_dirname = os.path.join('data', 'dataset', 'airqa', 'papers')
            for conference in os.listdir(pdf_dirname):
                pdf_filename = os.path.join(pdf_dirname, conference, f'{self.paper_id}.pdf')
                if os.path.exists(pdf_filename):
                    break
            else:
                return Observation(f'[Error]: paper id {self.paper_id} does not exist.')
            try:
                with open(pdf_filename, 'rb') as fin:
                    pdf_reader = PdfReader(fin)
                    mediabox = pdf_reader.pages[self.page_number - 1].mediabox
                    w, h = mediabox.width, mediabox.height
                image = convert_from_path(pdf_filename)[self.page_number - 1]
                width_ratio, height_ratio = image.width / w, image.height / h
            except IndexError:
                return Observation(f'[Error]: page {self.page_number} of paper id {self.paper_id} does not exist.')
            except Exception as e:
                return Observation(f'[Error]: {str(e)}')
        elif env.dataset == 'pdfvqa':
            pdf_page_dirname = os.path.join('data', 'dataset', 'pdfvqa', 'processed_data', 'test_images')
            if not any(fn.startswith(f'{self.paper_id}.pdf') for fn in os.listdir(pdf_page_dirname)):
                return Observation(f'[Error]: paper id {self.paper_id} does not exist.')
            pdf_page_filename = os.path.join(pdf_page_dirname, f'{self.paper_id}.pdf_{self.page_number - 1}.png')
            if not os.path.exists(pdf_page_filename):
                return Observation(f'[Error]: page {self.page_number} of paper id {self.paper_id} does not exist.')
            try:
                image = Image.open(pdf_page_filename)
                width_ratio = height_ratio = 1
            except Exception as e:
                return Observation(f'[Error]: {str(e)}')
        elif env.dataset == 'tatdqa':
            pdf_filename = os.path.join('data', 'dataset', 'tatdqa', 'processed_data', 'test_docs', f'{self.paper_id}.pdf')
            if not os.path.exists(pdf_filename):
                return Observation(f'[Error]: paper id {self.paper_id} does not exist.')
            try:
                with open(pdf_filename, 'rb') as fin:
                    pdf_reader = PdfReader(fin)
                    mediabox = pdf_reader.pages[self.page_number - 1].mediabox
                    w, h = mediabox.width, mediabox.height
                image = convert_from_path(pdf_filename)[self.page_number - 1]
                width_ratio, height_ratio = image.width / w, image.height / h
            except IndexError:
                return Observation(f'[Error]: page {self.page_number} of paper id {self.paper_id} does not exist.')
            except Exception as e:
                return Observation(f'[Error]: {str(e)}')
        else:
            raise ValueError(f'Unsupported dataset: {env.dataset}')

        try:
            if self.bounding_box:
                box = copy.deepcopy(self.bounding_box)
                box[2] = (box[0] + box[2]) * width_ratio
                box[3] = (box[1] + box[3]) * height_ratio
                box[0] *= width_ratio
                box[1] *= height_ratio
                image = image.crop(box)
            image_file = tempfile.mktemp(suffix='.png', dir=os.path.join(os.getcwd(), '.cache'))
            image.save(image_file, 'PNG')
            with open(image_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            os.remove(image_file)
            return Observation(image_data, 'image')
        except Exception as e:
            return Observation(f'[Error]: {str(e)}')
