import os
import sys
import warnings

from oauthenticator.generic import GenericOAuthenticator

c = get_config()  # noqa

# Oauth
c.JupyterHub.authenticator_class = GenericOAuthenticator
c.JupyterHub.services = [
    {
        'name': 'access-token-service',
        'admin': True,
        'url': 'http://127.0.0.1:8528',
        'command': [sys.executable, '/srv/jupyterhub/service.py'],
    }]
if 'JUPYTERHUB_CRYPT_KEY' not in os.environ:
    warnings.warn(
        "Need JUPYTERHUB_CRYPT_KEY env for persistent auth_state.\n"
        "    export JUPYTERHUB_CRYPT_KEY=$(openssl rand -hex 32)"
    )
    c.CryptKeeper.keys = [os.urandom(32)]
c.OAuthenticator.client_id = os.environ['KEYCLOAK_CLIENT']
c.OAuthenticator.scope = ["openid", "profile"]
c.OAuthenticator.client_secret = os.environ['KEYCLOAK_CLIENT_SECRET']
c.GenericOAuthenticator.login_service = os.environ['JUPYTERHUB_AUTH_NAME']
c.GenericOAuthenticator.enable_auth_state = True
c.GenericOAuthenticator.refresh_pre_spawn = True
keycloak_server = os.environ['KEYCLOAK_SERVER']
keycloak_realm = os.environ['KEYCLOAK_REALM']
c.GenericOAuthenticator.token_url = "{}/auth/realms/{}/protocol/openid-connect/token".format(keycloak_server,
                                                                                             keycloak_realm)
c.GenericOAuthenticator.authorize_url = "{}/auth/realms/{}/protocol/openid-connect/auth".format(keycloak_server,
                                                                                                keycloak_realm)
c.GenericOAuthenticator.userdata_url = "{}/auth/realms/{}/protocol/openid-connect/userinfo".format(keycloak_server,
                                                                                                   keycloak_realm)
c.GenericOAuthenticator.userdata_params = {'state': 'state'}
c.GenericOAuthenticator.username_key = "preferred_username"

if 'APPLICATION_NAME' in os.environ:
    c.DockerSpawner.environment = {'APPLICATION_NAME': os.environ["APPLICATION_NAME"]}

# launch with docker
c.JupyterHub.spawner_class = "docker"

# we need the hub to listen on all ips when it is in a container
c.JupyterHub.hub_ip = '0.0.0.0'
# the hostname/ip that should be used to connect to the hub
# this is usually the hub container's name
c.JupyterHub.hub_connect_ip = 'jupyterhub'

# pick a docker image. This should have the same version of jupyterhub
# in it as our Hub.
notebook_image = os.environ.get('DOCKER_NOTEBOOK_IMAGE')
c.DockerSpawner.image = notebook_image

# tell the user containers to connect to our docker network
network = os.environ.get('DOCKER_NETWORK_NAME') or 'jupyterhub'
c.DockerSpawner.network_name = network

# delete containers when the stop
c.DockerSpawner.remove = True
c.DockerSpawner.cmd = ["jupyter-labhub"]

# Explicitly set notebook directory because we'll be mounting a host volume to
# it.  Most jupyter/docker-stacks *-notebook images run the Notebook server as
# user `jovyan`, and set the notebook directory to `/home/jovyan/work`.
# We follow the same convention.
notebook_dir = os.environ.get('DOCKER_NOTEBOOK_DIR') or '/home/jovyan/work'
c.DockerSpawner.notebook_dir = notebook_dir
# Mount the real user's Docker volume on the host to the notebook user's
# notebook directory in the container
personal_notebooks_volume = os.environ.get('DATA_VOLUME_HOST') or '/opt/data'
personal_notebooks_volume += '/{}'
persisted_work = notebook_dir + '/persisted_work'
c.DockerSpawner.volumes = {personal_notebooks_volume.format("user-{username}"): persisted_work}

# For debugging arguments passed to spawned containers
c.DockerSpawner.debug = True
