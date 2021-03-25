import socket
import pymongo
import os
from flask import Flask,request,render_template,redirect
app=Flask(__name__,template_folder="C:/Users/sai revanth/OneDrive/Desktop")
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["mydatabase"]
check_in=db["check_in"]
from datetime import datetime
host = "0.tcp.ngrok.io"
port = int(11000)

# Create a TCP socket
while(True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect to the server with the socket via our ngrok tunnel
    server_address = (host, port)
    sock.connect(server_address)
    print("Connected to {}:{}".format(host, port))
    
    # Send the message
    message = "pls send data"
    print("Sending: {}".format(message))
    sock.sendall(message.encode("utf-8"))
    
    # Await a response
    data_received = 0
    data_expected = len(message)
    
    while data_received < data_expected:
        data = sock.recv(2048)
        data_received += len(data)
        today = datetime.now()
        print
        if(data):
            if(data.decode("utf-8")!="notfound"):
                db.check_in.insert_one({"employee":data.decode("utf-8"),"date and time":str(today)})
        
            print("Received: {}".format(data.decode("utf-8")))
    
            sock.close()
            break