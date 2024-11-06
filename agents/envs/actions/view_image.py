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
from pdf2image import convert_from_path


@dataclass
class ViewImage(Action):

    paper_id: str = field(default='', repr=True) # concrete paper id, required
    page_number: int = field(default=1, repr=True) # concrete page number, required
    bounding_box: List[float] = field(default_factory=list, repr=True) # concrete bounding box, optional

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the image.
        """
        if env.dataset == 'pdfvqa':
            image = Image.open(os.path.join('data', 'dataset', 'pdfvqa', 'processed_data', 'test_images', f'{self.paper_id}.pdf_{self.page_number - 1}.png'))
        elif env.dataset == 'tatdqa':
            image = convert_from_path(os.path.join('data', 'dataset', 'tatdqa', 'processed_data', 'test_docs', f'{self.paper_id}.pdf'))[self.page_number - 1]
        else:
            raise ValueError(f'Unsupported dataset: {env.dataset}')
        try:
            if self.bounding_box:
                box = copy.deepcopy(self.bounding_box)
                box[2] += box[0]
                box[3] += box[1]
                image = image.crop(box)
            image_file = tempfile.NamedTemporaryFile(suffix='.png', dir=os.path.join(os.getcwd(), '.cache'))
            image.save(image_file.name, 'PNG')
            with open(image_file.name, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            image_file.close()
            return Observation(image_data, 'image')
        except Exception as e:
            return Observation(f'[Error]: {str(e)}')

    @property
    def observation_type(self) -> str:
        return 'image'
