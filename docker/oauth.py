import os

import requests

JUPYTERHUB_API_TOKEN = os.getenv("JUPYTERHUB_API_TOKEN")
JUPYTERHUB_SERVICE_URL = "http://{}:8000/services".format(os.getenv("APPLICATION_NAME"))


def get_token():
    headers = {'Authorization': f'Token {JUPYTERHUB_API_TOKEN}'}
    url = f'{JUPYTERHUB_SERVICE_URL}/access-token-service/access-token'
    resp = requests.get(url, headers=headers)
    return resp.json().get('access_token')
