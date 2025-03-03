#coding=utf8
import os, json, re
from functools import cached_property
from typing import Dict, Any, Optional, List
from pymilvus import DataType
from utils.config import VECTORSTORE_DIR


def get_collection_name(
    embed_type: Optional[str] = None,
    embed_model: Optional[str] = None,
    modality: str = 'text',
    collection_name: Optional[str] = None
) -> str:
    """ Normalize the `collection_name` for Milvus vectorstore, or construct the `collection_name` from embedding configs automatically.
    """
    if collection_name is not None:
        return re.sub(r'[^a-z0-9_]', '_', collection_name.lower().strip())
    if modality in ['text', 'image']:
        collection_name = f"{modality}_{embed_type}_{os.path.basename(embed_model.rstrip(os.sep))}".lower()
        return re.sub(r'[^a-z0-9_]', '_', collection_name)
    raise NotImplementedError(f"Modality {modality} not supported yet.")


class VectorstoreField(object):
    
    __slots__ = ['name', 'dtype', 'is_primary', 'description', 'dim', 'max_length', 'element_type', 'max_capacity', 'auto_id']

    def __init__(self, field_obj: Dict[str, Any]):
        super(VectorstoreField, self).__init__()
        self.name: str = field_obj.get('name', 'field_name')
        dtype: str = field_obj.get('dtype', 'type')
        self.dtype = eval(f'DataType.{dtype}')
        self.is_primary: bool = field_obj.get('is_primary', False)
        self.description: str = field_obj.get('description', f"Field name: {field_obj['name']}; Data Type: {field_obj['dtype'].upper()}")

        if self.is_primary:
            assert self.dtype in [DataType.INT64, DataType.STRING, DataType.VARCHAR], f"Primary key field must be of type INT64, STRING or VARCHAR, but got {self.dtype}."
            self.auto_id: bool = True
        
        if '_VECTOR' in dtype and 'SPARSE' not in dtype: # for vector data type, specify the dimension
            assert 'dim' in field_obj, f"Dimension not specified for vector field {self.name}."
            self.dim: int = int(field_obj['dim'])
        
        if dtype in ['VARCHAR', 'STRING']: # for VARCHAR/STRING data type, specify the max_length
            self.max_length: int = int(field_obj.get('max_length', 65535))

        if dtype == 'ARRAY': # ARRAY DataType for bbox images
            etype: str = field_obj.get('etype', 'element_type').upper()
            self.element_type = eval(f"DataType.{etype}")
            self.max_capacity: int = int(field_obj.get('max_capacity', 20))
            if etype in ['VARCHAR', 'STRING']:
                self.max_length: int = int(field_obj.get('max_length', 256))


    def to_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.__slots__
                if hasattr(self, k) and 
                    getattr(self, k) is not None
        }


class VectorstoreIndex(object):

    __slots__ = ['field_name', 'index_type', 'index_name', 'metric_type', 'params']

    def __init__(self, index_obj: Dict[str, Any], field_type: str):
        super(VectorstoreIndex, self).__init__()
        default_index_type = 'FLAT' if 'VECTOR' in field_type and 'SPARSE' not in field_type \
            else 'SPARSE_INVERTED_INDEX' if 'VECTOR' in field_type and 'SPARSE' in field_type \
            else 'INVERTED'
        self.field_name: str = index_obj['field_name']
        self.index_type: str = index_obj.get('index_type', default_index_type)
        self.index_name: str = index_obj.get('index_name', f"{index_obj['field_name']}_index")

        if 'VECTOR' in field_type:
            default_metric_type = 'IP' if 'SPARSE' in field_type else 'COSINE'
            self.metric_type = index_obj.get('metric_type', default_metric_type)
        if 'params' in index_obj:
            self.params = index_obj['params']


    def to_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.__slots__
                if hasattr(self, k) and 
                    getattr(self, k) is not None
        }


class VectorstoreCollection(object):

    def __init__(self, collection_obj: Dict[str, Any]):
        super(VectorstoreCollection, self).__init__()
        self.modality: str = collection_obj['modality']
        assert self.modality in ['text', 'image'], f"Modality {self.modality} not supported yet."
        self.embed_type: str = collection_obj['embed_type']
        assert self.embed_type in ['sentence_transformers', 'bge', 'instructor', 'mgte', 'bm25', 'splade', 'clip'], f"Embedding type {self.embed_type} not supported yet."
        self.embed_model: str = collection_obj['embed_model']
        self.name: str = get_collection_name(
            modality=self.modality, embed_type=self.embed_type, embed_model=self.embed_model
        )
        if 'collection_name' in collection_obj:
            assert self.name == get_collection_name(collection_name=collection_obj['collection_name']), f"Collection name {collection_obj['collection_name']} does not follow the specified naming convention, it should be {self.name}."
        self.description: str = collection_obj.get('description', f"Vectorstore collection created from {self.modality} data with {self.embed_type} embedding model {self.embed_model}.")
        self.fields: List[VectorstoreField] = self._parse_fields(collection_obj.get('fields', []))
        self.indexes: List[VectorstoreIndex] = self._parse_indexes(collection_obj.get('indexes', []))


    def _parse_fields(self, fields: List[Dict[str, Any]]) -> List[VectorstoreField]:
        vs_fields = []
        for field_obj in fields:
            vs_fields.append(VectorstoreField(field_obj))
        return vs_fields


    def get_field_type(self, field_name: str) -> str:
        for field in self.fields:
            if field.name == field_name:
                return str(field.dtype)
        raise ValueError(f"Field {field_name} not found in collection {self.name}.")


    def _parse_indexes(self, indexes: List[Dict[str, Any]]) -> List[VectorstoreIndex]:
        vs_indexes = []
        for index_obj in indexes:
            vs_indexes.append(VectorstoreIndex(index_obj, self.get_field_type(index_obj['field_name'])))
        return vs_indexes



class VectorstoreSchema(object):


    def __init__(self, vectorstore: Optional[str] = None):
        super(VectorstoreSchema, self).__init__()
        self.vectorstore_path = os.path.join(VECTORSTORE_DIR, 'vectorstore_schema.json')
        if vectorstore is not None: # overwrite the general vectorstore path
            vs_folder = os.path.join(VECTORSTORE_DIR, vectorstore)
            if os.path.exists(os.path.join(vs_folder, 'vectorstore_schema.json')) and os.path.isfile(os.path.join(vs_folder, 'vectorstore_schema.json')):
                self.vectorstore_path = os.path.join(vs_folder, 'vectorstore_schema.json')
            elif os.path.exists(os.path.join(vs_folder, vectorstore + '.json')) and os.path.isfile(os.path.join(vs_folder, vectorstore + '.json')):
                self.vectorstore_path = os.path.join(vs_folder, vectorstore + '.json')
        self.vectorstore_schema: Dict[str, VectorstoreCollection] = self._load_schema()   


    def _load_schema(self) -> Dict[str, Any]:
        """ Load the vectorstore schema from the json file.
        @return: A dictionary of vectorstore schema.
            collection_name: {
            
            }
        """
        with open(self.vectorstore_path, mode='r', encoding='utf8') as inf:
            vs_schema = json.load(inf)
        vs_schema: Dict[str, VectorstoreCollection] = {
            get_collection_name(
                modality=collection_dict['modality'],
                embed_type=collection_dict['embed_type'],
                embed_model=collection_dict['embed_model']
            ): VectorstoreCollection(collection_dict)
            for collection_dict in vs_schema
        }
        return vs_schema


    def get_schema(self) -> Dict[str, VectorstoreCollection]:
        return self.vectorstore_schema


    @cached_property
    def collections(self) -> List[str]:
        return sorted(self.vectorstore_schema.keys())


    def get_collection(self, collection_name: str) -> VectorstoreCollection:
        return self.vectorstore_schema[collection_name]