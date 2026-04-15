import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from AirsimServer.unity_socket_server import UnitySocketServer


class UnitySocketServerTests(unittest.TestCase):
    def test_create_server_socket_returns_listening_socket(self):
        server = UnitySocketServer(host="localhost", port=0)
        sock = server._create_server_socket()
        try:
            self.assertIn(sock.family, (server.socket_family_ipv4, server.socket_family_ipv6))
            address = sock.getsockname()
            self.assertTrue(address[1] >= 0)
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
