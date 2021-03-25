import pymongo
import os
from flask import Flask,request,render_template,redirect
import socket
from datetime import datetime
app=Flask(__name__,template_folder="/")
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["mydatabase"]
check_in=db["check_in"]
@app.route('/login',methods=["GET","POST"])
def check():
    if request.method=="GET":
        return render_template('login.html')
    elif request.method=="POST":
        username=request.form["username"]
        password=request.form["password"]
        for document in db.admin.find():
            username_check=document["username"]
            password_check=document["password"]
            print("hi",username_check,password_check)
            if(username_check==username and password_check==password):
               return redirect("/query")
            
        return render_template("login.html",message="incorrect id or passsword") 
@app.route('/query',methods=["GET","POST"])
def get():
    if(request.method=="GET"):
        return render_template("after_login.html")
    elif request.method=="POST":
        l=[]
        s=f'for doc in {request.form["query"]}:l.append(doc)'
        print(s)
        try:
            exec(s)
        except:
            return "query executed successfully.database has been modified"
        #print(table)
        return render_template('after_login.html',message=l)
@app.route('/get_images')
def home():

 image_names = os.listdir('C:/Users/sai revanth/OneDrive/Desktop/dataset')
 image_names='C:/Users/sai revanth/OneDrive/Desktop/dataset/'+image_names
 print(image_names)
 return render_template("images.html", image_names1=image_names)
app.run()
def put():
    s=""
    file_path="C:/Users/sai revanth/OneDrive/Desktop/dataset"
    for file_name in os.listdir(file_path) :
      #  print(file_name+"    ")
        s=s+"<img src='"+file_path+"/"+file_name+"'>"   
        
        
    print(s)
    
    return render_template("jiij.html", message=s)
app.run()