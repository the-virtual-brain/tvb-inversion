import json
import logging
import os

import tornado.ioloop
from jupyterhub.services.auth import HubAuthenticated
from tornado.httpclient import AsyncHTTPClient
from tornado.web import RequestHandler, Application, authenticated


# This service is copied from https://github.com/HumanBrainProject/jupyterhub-access-token-service


class AccessTokenHandler(HubAuthenticated, RequestHandler):
    ''' Respond to Access Token requests. '''

    async def _request_token(self, username: str) -> str:
        user_endpoint = f'{self.hub_auth.api_url}/users/{username}'
        headers = {
            'Authorization': f'token {self.hub_auth.api_token}',
        }
        client = AsyncHTTPClient()
        resp = await client.fetch(user_endpoint, headers=headers)
        return json.loads(resp.body)

    @authenticated
    async def get(self):
        user = self.get_current_user()
        logging.debug(user)
        user_json = await self._request_token(user['name'])
        logging.debug(user_json)
        self.write({'access_token': user_json['auth_state']['access_token']})


def main():
    prefix = os.environ.get('JUPYTERHUB_SERVICE_PREFIX', '')
    app = Application([
        (prefix + 'access-token', AccessTokenHandler),
    ])
    app.listen(8528)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
