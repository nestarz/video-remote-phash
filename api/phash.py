from http.server import BaseHTTPRequestHandler
from urllib import parse

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("videohash")

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from videohash import VideoHash

        s = self.path
        dic = dict(parse.parse_qsl(parse.urlsplit(s).query))
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        if "url" in dic:
            message = str(VideoHash(url=dic["url"]))
        else:
            message = "Missing url!"
        self.wfile.write(message.encode())
        return