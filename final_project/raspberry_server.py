import sys
import socket
import select
import random
import time
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

TRIG = 4 
ECHO = 17

print "Distance Measurement In Progress"

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

GPIO.output(TRIG, False)
print "Waiting For Sensor To Settle"
time.sleep(2)

HOST = ''
SOCKET_LIST = []
RECV_BUFFER = 1024
PORT = 9009


def chat_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)

    # add server socket object to the list of readable connections
    SOCKET_LIST.append(server_socket)

    print "Chat server started on port " + str(PORT)

    while True:

        # get the list sockets which are ready to be read through select
        # 4th arg, time_out  = 0 : poll and never block
        ready_to_read, ready_to_write, in_error = select.select(SOCKET_LIST, [], [], 0)

        for sock in ready_to_read:
            # a new connection request recieved
            if sock == server_socket:
                sockfd, addr = server_socket.accept()
                SOCKET_LIST.append(sockfd)
                print "Client (%s, %s) connected" % addr

            # a message from a client, not a new connection
            else:
                # process data recieved from client,
                try:
                    # receiving data from the socket.
                    data = sock.recv(RECV_BUFFER)
                    if data:
                        # there is something in the socket
                        # broadcast(server_socket, sock, "\r" + '[' + str(sock.getpeername()) + '] ' + data)
                        GPIO.output(TRIG, True)
                        time.sleep(0.00001)
                        GPIO.output(TRIG, False)

                        while GPIO.input(ECHO)==0:
                            pulse_start = time.time()

                        while GPIO.input(ECHO)==1:
                            pulse_end = time.time()


                        pulse_duration = pulse_end - pulse_start

                        distance = pulse_duration * 17150

                        distance = round(distance, 1)
                        sock.send(str(distance))
                        # time.sleep(0.001)
                    else:
                        # remove the socket that's broken
                        if sock in SOCKET_LIST:
                            SOCKET_LIST.remove(sock)

                        # exception
                except:
                    continue

    server_socket.close()

if __name__ == "__main__":
    sys.exit(chat_server())

