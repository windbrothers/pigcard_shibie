# coding = utf-8
import requests
import json
def login():
     host = "http://172.15.4.62:8000"
     endpoint = r"/base/login/appLogin"

     url = ''.join([host, endpoint])
     headers = \
          {
        "Content-Type": "application/json;charset=UTF-8"
             }
     body = \
     {
        "account": "designer",
        "password": "designer",
        "appId": "base"
        }
     r = requests.post(url, headers=headers, data=json.dumps(body))
     context=r.text
     tokenString=context.split(",")[-4]
     pwd=tokenString.split("\"")[-2]
     #pwd=context.[1]

     return pwd

