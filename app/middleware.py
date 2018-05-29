import json
import os

from werkzeug.wrappers import Response

FACEREC_API_TOKEN = os.environ.get('FACEREC_API_TOKEN', 'facerec_api_dev_token')

class TokenRequiredMiddleware(object):

    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        request_api_token = environ.get('HTTP_AUTHENTICATION')
        
        if request_api_token != FACEREC_API_TOKEN:
            code = 401
            response = json.dumps({'status': code, 'error': 'FACEREC_API Token Required'})
            headers = [('Content-Type', 'application/json')]
            return Response(response, code, headers)(environ, start_response)

        return self.app(environ, start_response)
