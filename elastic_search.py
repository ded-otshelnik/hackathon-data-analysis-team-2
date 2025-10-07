import json

from tqdm import tqdm

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import warnings
warnings.filterwarnings("ignore")

def read_json(file_path: str) -> dict:
    """Read json file and convert data to dict

    Args:
        file_path (str): path to json file

    Returns:
        data (dict): DataFrame with data from json file
    """
    with open(file_path, "r", encoding = 'utf-8') as file:
        return json.load(file)
    
def create_index(agent: Elasticsearch, index_name: str, settings: dict) -> None:
    """Create index in Elasticsearch

    Args:
        agent (Elasticsearch): agent for interaction with Elasticsearch
        index_name (str): name of index
        settings (dict): settings for index
    """
    try:
        agent.indices.create(index=index_name, settings=settings)
    except:
        print(f"Index {index_name} already exists")

def upload(index, data):
    """Upload data to Elasticsearch

    Args:
        index: index in Elasticsearch
        data: data for uploading to Elasticsearch

    Yields:
      data by index
    """
    for item in tqdm(data, total=len(data), desc="Uploading data to Elasticsearch"):
        yield {
            "_index": index,
            "_source": item
        }
    

if __name__ == "__main__":
    # authorization data
    login = 'elastic'
    password = 'jinr_elastic'
    index = 'hackathon_articles'

    # connect to Elasticsearch server
    url = 'http://localhost:9200'
    es = Elasticsearch(
        url,
        basic_auth=(login, password),
        verify_certs=False
    )

    # create index by settings from json file
    settings = read_json('data/mapping.json')
    create_index(es, index, settings)

    # upload data to Elasticsearch created index
    data = read_json('data/articles.json')
    bulk(es, upload(index, data['articles']))