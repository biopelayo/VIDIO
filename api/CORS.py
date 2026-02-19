import falcon


class CORSComponent:
    def process_response(self, req, resp, resource, req_succeeded):
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        resp.set_header('Access-Control-Allow-Headers',
                        'Authorization, Content-Type, Accept, Origin, X-Requested-With')
        resp.set_header('Access-Control-Max-Age', '86400')

    def process_request(self, req, resp):
        if req.method == 'OPTIONS':
            resp.status = falcon.HTTP_200
            raise falcon.HTTPStatus(falcon.HTTP_200)
