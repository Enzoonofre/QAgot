'''This module contains utility functions for the project'''
import mmh3


def get_unique_docs(dataset, unique_docs):
    '''Get unique documents from dataset
    
    Args:
    dataset: list of dictionaries
    unique_docs: set of document ids

    Returns:
    docs: list of dictionaries
    '''
    docs = list()
    for i in dataset:
        if i["context"] is not None:
            document_id = mmh3.hash128(i["context"], signed=False)
            if document_id not in unique_docs:
                unique_docs.add(document_id)

                document = {}
                document['content'] = i['context']
                document['id'] = document_id
            
                aux = {}
                aux['context_title'] = i['context_title']
                aux['context.id'] = i['context_id']
                document['meta'] = aux

                docs.append(document)
    return docs
