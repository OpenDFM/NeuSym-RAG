#coding=utf8
from abc import ABC, abstractmethod


class BaseOutputParser(ABC):

    @abstractmethod
    def parse(self, output: str) -> dict:
        """ Parse the output from the model into structured format.
        """
        pass