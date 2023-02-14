import os
from flask import Flask, request, render_template


url= "" #url表示用户选择文件的地址，即需要处理的视频地址
url1= "" #url1表示处理完成后视频的地址

app = Flask(__name__,
            static_folder='./static', #设置静态和模板文件夹
            template_folder='./templates')

# 根路由，返回首页
@app.route('/')
def hello():
    return render_template('home.html')

# 设置允许的文件格式、大小
# 由于html允许的视频格式为mp4，因此我们设置仅有mp4的文件能进行上传
ALLOWED_EXTENSIONS = set(['mp4'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# 添加首页路由
@app.route('/home')
def hello_home():
    return render_template('home.html')

#添加关于项目页面路由
@app.route('/about')
def hello_about():
    return render_template('about.html')

#添加项目展示页面路由
@app.route('/show', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file') #拿到用户上传的文件
        filelist = []
        for file in files:
            filename = file.filename #找到文件的名字，以及文件类型
            filetype = filename.split('.')[-1]
            # 有文件夹则添加，没有即自动创建
            uploadpath = os.getcwd() + os.sep + 'static/videos'
            if not os.path.exists(uploadpath):
                os.mkdir(uploadpath)

            #将用户上传的视频文件保存到上述创建的路径下
            file.save(uploadpath + os.sep + filename)
            filelist.append(filename)
            # 视频回显url
            global url
            url = "./static/videos/"+filename

        #返回项目展示页面，并向前端传递相关数据
        return render_template('show.html', msg='文件上传成功!', filelist=filelist, url=url)
    else:
        return render_template('show.html', msg='等待上传...', url="")

#定义一个路由，方便与前端表单链接
@app.route('/handle', methods=['GET'])
def handle():

    ############################################################
    ####################       视频处理        ##################
    #1.在这里进行视频的处理，url指的是需要处理的视频地址
    #2.url1是指处理完成视频的地址
    global url;
    global url1;

    ##############################################################
    #返回项目展示页面，并向前端传递相关数据
    return render_template('show.html', msg='文件上传成功!', msg1='处理完成!', url=url ,url1=url1)


if __name__ == '__main__':
    app.run(port=8000, debug=False)