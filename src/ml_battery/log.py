from logging import *
import logging.handlers
import socket

class SocketHandler(logging.handlers.SocketHandler):
    def makeSocket(self, timeout=1):
        result = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result.connect(self.address)
        except OSError:
            result.close()
            raise
        return result

rootLogger = getLogger('')
rootLogger.setLevel(DEBUG)
socketHandler = SocketHandler('localhost',
                    logging.handlers.DEFAULT_TCP_LOGGING_PORT)
rootLogger.addHandler(socketHandler)
