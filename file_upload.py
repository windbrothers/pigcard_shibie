import requests

def up(src,pwd):
     A="Bearer"
     PWD=A+' '+pwd
     url = 'http://172.15.4.62:8000/file/api/dfs/upload'
     headers = \
         {
             # "Content-Type": "multipart/form-data",
#             "Authorization": "Bearer eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJkZXNpZ25lckBAQCIsImlhdCI6MTU4NzQzOTExOSwiZXhwIjoxNTg3NDQyNzE5fQ.B1kBTDif5OHBM29XpldE4aEk2lckFWclPytF9flEfx3E2lhPgJpQpLHjpZuGtodFu1JgGZZJr69SC0FslMV_Lg"
             "Authorization": PWD
         }
     files = {'file': open(src, 'rb')}
#     files = {'file': open('demo.jpg', 'rb')}
     options = {'path': 'iot', 'scene': 'iot'}  # 参阅浏览器上传的选项
     r = requests.post(url, data=options, files=files, headers=headers)
     context=r.text
#     print(context)
     tokenString=context.split(",")[-3]
     msg=tokenString.split("\"")[-2]
     name=msg.split("/")[-1]
     return name

