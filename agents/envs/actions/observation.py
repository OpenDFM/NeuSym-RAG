#coding=utf8

import base64
from typing import Dict, Any


class Observation():

    def __init__(self, obs_content: str, obs_type: str = 'text'):
        """
        @param:
            obs_content: str, the text content or image path of the observation.
            obs_type: str, the observation type should be one of ["text", "image"], default is "text".
        """
        self.obs_content = obs_content
        self.obs_type = obs_type
        assert self.obs_type in ['text', 'image'], f'Invalid observation type: {self.obs_type}'


    def convert_to_message(self) -> Dict[str, Any]:
        if self.obs_type == 'text':
            msg_content = f'[Observation]:\n{self.obs_content}' if '\n' in str(self.obs_content) else f'[Observation]: {self.obs_content}'
        elif self.obs_type == 'image':
            try:
                with open(self.obs_content, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                msg_content = [
                    {
                        'type': 'text',
                        'text': '[Observation]: The extracted image is shown below.'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': 'data:image/jpeg;base64,' + image_data
                        }
                    }
                ]
            except Exception as e:
                return Observation(f'[Error]: {str(e)}').convert_to_message()
        else:
            raise ValueError('Invalid observation type: ' + self.obs_type)
        return {'role': 'user', 'content': msg_content}
