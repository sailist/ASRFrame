from tkinter import *
from scipy.io import wavfile
from matplotlib import pyplot as plt
from tkinter import filedialog
import tkinter.messagebox
import os
import wave
import numpy as np
import _thread
import time
import visualization.dcb as dcb
import  ffmpeg
from pydub import AudioSegment
import jointly.DCHMM
from core.base_model import BaseJoint

dir_path = os.path.split(os.path.realpath(__file__))[0] #"./util"

class Front():
    state = 0                           #录音状态 1为录音中
    tagnum = 0                          #标签数
    soundfile = os.path.join(dir_path,"cache.wav")              #声音文件
    allstr = ''                          #全部字符
    getstr = ''                          #获取字符
    currentstr = ''                    #当前字符串
    restr = ''                    #替换字符
    engstr=[]                       #拼音字符
    engerr=[]                        #拼音错误位置
    app = Tk()
    text1 = Text(app, width=50, height=3)
    text2 = Text(app, width=50, height=4)
    rere = -1                        #当前错误指标
    num = 2                         #总错误数
    enderr =[]                        #可能错误
    enderr2=[]                         #变动错误
    curerr=[0,0]                          #当前错误
    repstate = 0                       #替换状态
    messsage = StringVar()
    def create(self, model:BaseJoint):
        self.model = model

        self.app.title('语音识别')
        self.pr = dcb.proc()
        
        frm0 = Frame(width=200, height=50, bg='white')
        frm0.grid(row=0, column=0)
        pp1 = PhotoImage(file = os.path.join(dir_path,"imgs1.png"))
        pp2 = PhotoImage(file = os.path.join(dir_path,"imgs2.png"))
        pp3 = PhotoImage(file = os.path.join(dir_path,"imgs3.png"))
        pp4 = PhotoImage(file = os.path.join(dir_path,"imgs4.png"))
        pp5 = PhotoImage(file = os.path.join(dir_path,"imgs5.png"))
        pp6 = PhotoImage(file = os.path.join(dir_path,"imgs6.png"))
        pp7 = PhotoImage(file = os.path.join(dir_path,"imgs7.png"))
        pp8 = PhotoImage(file = os.path.join(dir_path,"imgs8.png"))
        pp9 = PhotoImage(file = os.path.join(dir_path,"imgs9.png"))
        pp10 = PhotoImage(file = os.path.join(dir_path,"imgs10.png"))
        button1 = Button(frm0,text = "录音",command = self.reco,image=pp1)
        button2 = Button(frm0, text = "暂停", command = self.pause, image=pp2)
        button3 = Button(frm0, text = "继续", command = self.goon, image=pp3)
        button4 = Button(frm0, text = "完成", command = self.finish, image=pp4)
        button5 = Button(frm0,text = "选择",command = self.selectfile,image=pp5)
        button6 = Button(frm0,text = "识别",command = self.reconfile,image=pp6)
        button7 = Button(frm0,text = "重录",command = self.rerew,image=pp7)
        button8 = Button(frm0,text = "重读替换",command = self.rep,image=pp8)
        button9 = Button(frm0,text = "确认",command = self.asure,image=pp9)
        button10 = Button(frm0,text = "重置",command = self.restart,image=pp10)
        button1.grid(row=0,column=0)
        # button2.grid(row=0,column=1)
        # button3.grid(row=0,column=2)
        button4.grid(row=0,column=3)
        button5.grid(row=0,column=4)
        button6.grid(row=0,column=5)
        button7.grid(row=0,column=6)
        button8.grid(row=0,column=7)
        button9.grid(row=0,column=8)
        button10.grid(row=0,column=9)
        self.mod1 = [
            ("ModelA",1),
            ("ModelB",2),
            ("ModelC",3),
            ("ModelD",4),
        ]
        self.mod2 = [
            ("ModelA",1),
            ("ModelB",2),
            ("ModelC",3),
            ("ModelD",4),
        ]
        
        frm1 = Frame(width=200, height=50, bg='white')
        frm2 = Frame(width=200, height=50, bg='white')
        frm1.grid(row=1, column=0)
        frm2.grid(row=2, column=0)
        Label(frm1,text="语音模型").grid(row=0,column=0)
        v1=StringVar()
        for text, mode in self.mod1:
            b = Radiobutton(frm1, text=text,variable=v1, value=mode)
            b.grid(row=0,column=mode)
        Label(frm2,text="语言模型").grid(row=0,column=0)
        v2=StringVar()
        for text, mode in self.mod2:
            b = Radiobutton(frm2, text=text,variable=v2, value=mode)
            b.grid(row=0,column=mode)

        self.text1.grid(row=3, column=0)
        self.text2.grid(row=4, column=0)
        self.messsage.set('欢迎使用语音识别')
        Label(self.app,textvariable = self.messsage).grid(row=5,column=0)
        
        self.app.mainloop()
        
        
        
#MP3转wav        
    def change(self):
        AudioSegment.converter = os.path.join(dir_path,"ffmpeg.exe")
        song = AudioSegment.from_mp3(self.soundfile)
        song.export(os.path.splitext(self.soundfile)[0]+".wav",format="wav")
        self.soundfile = os.path.splitext(self.soundfile)[0]+".wav"
        
#选择文件        
    def selectfile (self):
        self.messsage.set('选择中...')
        fpath = filedialog.askopenfilename()
        self.soundfile = fpath
        print(self.soundfile)
        if self.soundfile=="":
            self.soundfile=os.path.join(dir_path,"cache.wav")
        if os.path.splitext(self.soundfile)[-1] == ".mp3":
            self.change()
        self.messsage.set('已完成')

    def testfun(self,fdk,arg1):
        def fun(fdk):
            for i in range(self.tagnum):
                if i == arg1:
                    self.text1.tag_config('tag'+str(arg1),background = 'blue',foreground = 'yellow')
                    self.rere = arg1
                    self.curerr[0]=self.enderr2[2*arg1]
                    self.curerr[1]=self.enderr2[2*arg1+1]
                    self.currentstr = self.allstr[self.enderr[2*self.rere]:self.enderr[2*self.rere+1]]
                    print(self.currentstr)
                else :
                    self.text1.tag_config('tag'+str(i),background = 'white',foreground = 'red')
        return fun            
    
#识别    
    def reconfile (self):
        if self.pr.ending ==0 :
            tkinter.messagebox.showinfo('注意','正在录音中...')
        else :
            suf=self.soundfile
            self.messsage.set('识别中...')
            fr,wave_data = wavfile.read(suf)
            plt.plot(wave_data)
            plt.show()
            enst,self.getstr,apstr  =self.model.raw_record(wave_data)             #获取拼音串
            enno=apstr               #获取拼音错误值

            curerr=[0,0]                                     #当前错误值重置
            # self.num = int(len(apstr)/2)                            #错误数
            self.num = 0#int(len(apstr)/2)                            #错误数
            self.tagnum = self.tagnum+self.num                      #总标签数
            # for i in range(len(enno)):
            #     enno[i]=enno[i]+len(self.engstr)
            self.engstr=self.engstr+enst
            for i in range(len(apstr)):
                apstr[i]=apstr[i]+len(self.allstr)
            self.enderr=self.enderr+apstr
            self.enderr2=self.enderr2+apstr
            self.allstr=self.allstr+self.getstr
            self.text1.insert(END,self.getstr)
            for arg in range(self.num):
                self.text1.tag_add('tag'+str(self.tagnum-self.num+arg),'1.'+str(self.enderr[2*arg+2*(self.tagnum-self.num)]),'1.'+str(self.enderr[2*arg+1+2*(self.tagnum-self.num)]))
                self.text1.tag_config('tag'+str(self.tagnum-self.num+arg),background = 'white',foreground = 'red')
                self.text1.tag_bind('tag'+str(self.tagnum-self.num+arg),'<Button-1>',self.testfun(self,self.tagnum-self.num+arg))
            stt=''
            for i in range(len(enst)):
                stt=stt+self.engstr[i]+' '
            self.text2.insert(END,stt)    

            self.pr.recc()                          #清空frames
            self.messsage.set('已完成')

    #录音
    def reco(self):
        if self.state == 0 :
            try:
                self.pr.filepath = os.path.join(dir_path,"cache.wav")
                _thread.start_new_thread(self.pr.get_audio , (self.pr,) )
                self.state = 1
                self.messsage.set('正在录入语音...')
            except:
                print ("Error: unable to start thread")
        else :
            tkinter.messagebox.showinfo('注意','正在录音中...')

    #暂停
    def pause(self):
        self.pr.cut_audio()
        self.messsage.set('已暂停')
        
    #继续
    def goon(self):
        self.pr.cont_audio()
        self.messsage.set('正在录入语音...')
#完成
    def finish(self) :
        self.pr.fini_audio()
        self.soundfile = os.path.join(dir_path,"cache.wav")
        while self.pr.ending == 0:
            pass
        self.state = 0
        self.messsage.set('已完成')


#重读        
    def rerew(self) :
        self.rere = 1
        self.pr.filepath = os.path.join(dir_path,"cache2.wav")
        self.soundfile = os.path.join(dir_path,"cache2.wav")
        self.reco()
#重读替换
    def rep(self):
        if self.pr.ending ==0 :
            tkinter.messagebox.showinfo('注意','正在录音中...')
        elif self.curerr==[0,0]:
            tkinter.messagebox.showinfo('注意','未选中')
        else :
            if self.rere !=-1 :
                self.messsage.set('修正中...')
                f = wave.open(os.path.join(dir_path,"cache2.wav"), "rb")
                params = f.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                str_data = f.readframes(nframes)
                f.close()
                wave_data = np.fromstring(str_data, dtype=np.short)
                self.restr,enstrr,er2  =self.model.raw_record(wave_data)         #替换数据
                             
                self.engstr=self.engstr[0:self.curerr[0]]+enstrr+self.engstr[self.curerr[1]:]
                self.text2.delete('1.0',END)
                stt=''
                for i in range(len(self.engstr)):
                    stt=stt+self.engstr[i]+' '
                self.text2.insert(END,stt)    
                
                
                
                self.text1.delete('1.'+str(self.curerr[0]),'1.'+str(self.curerr[1]))
                self.text1.insert('1.'+str(self.curerr[0]),self.restr)

                self.text1.tag_add('tag00','1.'+str(self.curerr[0]),'1.'+str(self.curerr[0]+len(self.restr)))
                print(self.curerr[0]+len(self.restr))
                self.text1.tag_config('tag00',background = 'yellow',foreground = 'blue')
                self.curerr[1]=self.curerr[1]+len(self.restr)-len(self.currentstr)
                for i in range(2*(self.tagnum-self.rere)-1):
                    self.enderr2[2*self.rere+i+1]=self.enderr2[2*self.rere+i+1]+len(self.restr)-len(self.currentstr)
                self.currentstr=self.restr
                self.repstate = 1
                
            print(self.enderr2)
#确认
    def asure(self):
        if self.repstate == 0:
            self.text1.tag_delete('tag'+str(self.rere))
        else :
            self.text1.tag_delete('tag00')
        self.messsage.set('已完成')
        self.rere=-1
        self.repstate = 0
        self.curerr=[0,0]
#重置
    def restart(self):
            self.state = 0
            self.tagnum = 0
            self.soundfile = os.path.join(dir_path,"cache.wav")
            self.allstr = ''
            self.getstr = ''
            self.currentstr = ''                    
            self.restr = ''
            self.engstr=[]
            self.engerr=[] 
            self.rere = -1                        
            self.num = 0                         
            self.enderr =[]                        
            self.enderr2=[]                         
            self.curerr=[0,0]                          
            self.pr.sign = 0
            self.pr.fini = 0
            self.pr.rereco = 0
            self.pr.ending = 0
            self.pr.filepath = os.path.join(dir_path,"cache.wav")
            self.text1.delete('1.0',END)
            self.text2.delete('1.0',END)
            self.messsage.set('欢迎使用语音识别')