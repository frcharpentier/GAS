from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer #, HTTPServer
import logging
import os
from os.path import getsize

class EXEC_ADDRESS:
    @staticmethod
    def test(chemin):
        return False
    
    def __init__(self, chemin, rqHandler, wfile, headers=None, rfile=None):
        self.chemin = chemin
        self.rqHandler = rqHandler
        self.wfile = wfile
        self.headers = headers
        self.rfile = rfile

    def sendFile(self, ctype, fichier):
        if not os.path.exists(fichier):
            fichier = os.path.join(os.path.dirname(os.path.abspath(__file__)), fichier)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        taille = getsize(fichier)
        self.send_header("Content-Length", "%d"%taille)
        self.end_headers()
        with open(fichier, "rb") as F:
            self.wfile.write(F.read())

    def send_response(self, *args, **kwargs):
        self.rqHandler.send_response(*args, **kwargs)

    def send_header(self, *args, **kwargs):
        self.rqHandler.send_header(*args, **kwargs)

    def end_headers(self):
        self.rqHandler.end_headers()

    def execute(self):
        pass


class ServeurReq(BaseHTTPRequestHandler):

    liste_pour_get = []
    liste_pour_post = []

    @classmethod
    def add_post(cls, *args):
        assert all(issubclass(x, EXEC_ADDRESS) for x in args)
        cls.liste_pour_post.extend(args)

    @classmethod
    def add_get(cls, *args):
        assert all(issubclass(x, EXEC_ADDRESS) for x in args)
        cls.liste_pour_get.extend(args)

    def do_GET(self):
        chemin = self.path
        trouve = False
        for x in self.liste_pour_get:
            if x.test(chemin):
                trouve = True
                resp = x(chemin, self, self.wfile)
                resp.execute()
                break
        if not trouve:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        chemin = self.path
        trouve = False
        for x in self.liste_pour_post:
            if x.test(chemin):
                trouve = True
                resp = x(chemin, self, self.wfile, self.headers, self.rfile)
                resp.execute()
                break
        if not trouve:
            self.send_response(404)
            self.end_headers()

        
def lancer_serveur(nom_hote, num_port):        
    #serveurHTTP = HTTPServer((nom_hote, num_port), ServeurReq)
    serveurHTTP = ThreadingHTTPServer((nom_hote, num_port), ServeurReq)
    logging.info("Serveur démarré : http://%s:%s\n" % (nom_hote, num_port))

    try:
        serveurHTTP.serve_forever()
    except KeyboardInterrupt:
        pass

    serveurHTTP.server_close()
    logging.info("Server stopped.\n")