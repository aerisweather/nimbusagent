import logging
from typing import Union, List

import openai
from numpy import dot
from numpy.linalg import norm

FUNCTIONS_EMBEDDING_MODEL = "text-embedding-ada-002"


def is_query_safe(query: str) -> bool:
    """Returns True if the query is considered safe, False otherwise."""
    response = openai.Moderation.create(input=query)
    result = response.get('results', [{}])[0]

    if result.get('flagged', False):
        logging.debug(f"Query '{query}' was flagged by OpenAI's moderation API. {result}")
        return False

    return True


def get_embedding(text, model=FUNCTIONS_EMBEDDING_MODEL):
    try:
        text = text.replace("\n", " ")
        embedding = openai.Embedding.create(
            input=text,
            model=model
        )["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def cosine_similarity(List1, List2):
    """ get cosine similarity of two vector of same dimensions """
    return 1 - dot(List1, List2) / (norm(List1) * norm(List2))


def find_similar_embedding_list(query: str, function_embeddings: list, k_nearest_neighbors: int = 1):
    """Return the k function descriptions most similar (least cosine distance) to given query"""
    if not function_embeddings:
        return None

    query_embedding = get_embedding(query)

    distances = []
    for function_embedding in function_embeddings:
        dist = cosine_similarity(query_embedding, function_embedding['embedding'])
        distances.append(
            {'name': function_embedding['name'], 'distance': dist})

    sorted_distances = sorted(distances, key=lambda x: x['distance'])

    return sorted_distances[:k_nearest_neighbors]


def combine_lists_unique(list1: List, set2: Union[List, set]) -> List[any]:
    new_list = list1.copy()
    for item in set2:
        if item not in new_list:
            new_list.append(item)
    return new_list
