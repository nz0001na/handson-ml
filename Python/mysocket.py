import socket

mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect(('data.pr4e.org', 80))

cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
mysock.send(cmd)

while True:
    data = mysock.recv(512)
    if len(data) < 1:
        break
    print(data.decode())
mysock.close()
#
# # Send data to the server
# message = b'Hello, server!'
# sock.sendall(message)
#
# # Receive response from the server
# data = sock.recv(1024)
# print("Server response:", data.decode())
#
# # Close the socket
# sock.close()