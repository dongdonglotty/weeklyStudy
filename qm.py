# -*- coding: UTF-8 -*-
import tkinter.ttk as ttk
from tkinter import LabelFrame
from tkinter import Label
from tkinter import StringVar
from tkinter import Entry
from tkinter import Button
from tkinter import Tk
from pandastable import Table
import tkinter.filedialog as tf
import tkinter.messagebox as tm
from pandas import DataFrame as daf
from pandas import read_csv


class App(object):
    def __init__(self,object):
        self.root = object
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        print (w)
        print (h)
        self.filename = StringVar()
        self.dataframe = daf()
        self.groupname = StringVar()
        self.gradename = StringVar()
        
        
        self.lb1=LabelFrame(object, width=w*0.6, height=h*0.7, text='Group Point Table')
        self.lb1.grid(row=0,column=0,rowspan=4,columnspan=4,padx=2,pady=2)
        self.lb1.grid_propagate(True)
        self.lb1.grid_rowconfigure(0, weight=1)
        self.pt = Table(self.lb1,width=w*2/3,height=h*0.7)
        self.pt.show()
        
        
        self.lb2 = LabelFrame(object, width=w*0.3, height=h*0.35, text='Content Edit')
        #self.lb3.grid(row=0, column=1,rowspan=2)
        self.lb2.grid(row=0,column=5,rowspan=2,columnspan=2,padx=2,pady=2)
        
        self.label2_1 = Label(self.lb2,text='Group:')
        self.label2_1.grid(row=0,column=0)

        self.groupname = StringVar()
        self.addgrade = StringVar()
        
        self.entry2_1 = Entry(self.lb2,textvariable=self.groupname)
        self.entry2_1.grid(row=0,column=1)
        
        def addgroup(df,gname,path):
            if gname not in (list(df['Group'])):
                df = df.append({'Group': gname}, ignore_index=True)
                self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
                self.dataframe = df
                pt.show()
                df.to_csv(path,index=False)
            else:
                tm.showinfo(title='WARNING', message='The group already exists!')
            
        def addgrade(df,classname,path):
            if classname not in (list(df.columns)):
                columns_list = df.columns.tolist()
                columns_list.insert(-1, classname)
                df = df.reindex(columns=columns_list, fill_value=0)
                self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
                self.dataframe = df
                print (df.columns)
                pt.show()
                df.to_csv(path,index=False)
            else:
                tm.showinfo(title='WARNING', message='The bonus point already exists!')
        
        def delgroup(df,gname,path):
            if gname in (list(df['Group'])):
                gindex = df[df['Group']==gname].index[0]
                df.drop(gindex,axis=0,inplace=True)
                self.dataframe = df
                self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
                pt.show()
                self.dataframe = df
                df.to_csv(path,index=False)
            else:
                tm.showinfo(title='WARNING', message='The group does not exist!')
        
        def delgrade(df,classname,path):
            if classname in (list(df.columns)):
                df.drop([classname],axis=1,inplace=True)
                self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
                self.dataframe = df
                pt.show()
                df.to_csv(path,index=False)
            else:
                tm.showinfo(title='WARNING', message='The bonus point does not exist!')
        
        self.button2_1_1 = Button(self.lb2,text='Add Group',command=lambda: addgroup(self.dataframe,self.entry2_1.get(),self.fentry.get()))
        self.button2_1_1.grid(row=1,column=0)
        
        self.button2_1_2 = Button(self.lb2,text='Clean All',command=lambda: delgroup(self.dataframe,self.entry2_1.get(),self.fentry.get()))
        self.button2_1_2.grid(row=1,column=1)
        
        self.label2_2 = Label(self.lb2,text='Bonus Point:')
        self.label2_2.grid(row=2,column=0)
        
        self.entry2_2 = Entry(self.lb2,textvariable=self.addgrade)
        self.entry2_2.grid(row=2,column=1)
        
        self.button2_2_1 = Button(self.lb2,text='Add Bonus Point',command=lambda: addgrade(self.dataframe,self.entry2_2.get(),self.fentry.get()))
        self.button2_2_1.grid(row=3,column=0)
        
        self.button2_2_2 = Button(self.lb2,text='Clean All',command=lambda: delgrade(self.dataframe,self.entry2_2.get(),self.fentry.get()))
        self.button2_2_2.grid(row=3,column=1)
    
        
        self.lb3 = LabelFrame(object, width=w*0.3, height=h*0.35, text='Group Evaluation')
        #self.lb3.grid(row=0, column=1,rowspan=2)
        self.lb3.grid(row=3,column=5,rowspan=2,columnspan=2,padx=2,pady=2)
        
        
        self.label3_1 = Label(self.lb3,text='Group:')
        self.label3_1.grid(row=0,column=0)
        self.label3_2 = Label(self.lb3,text='Bonus Point:')
        self.label3_2.grid(row=0,column=1)
        
        def upd_combobox_g():
            dClasses = self.dataframe['Group']
            self.box3_1['values'] = tuple(dClasses)
            
        def upd_combobox_c():
            df = self.dataframe
            df.set_index(['Group'],inplace=True)
            dColumns = df.columns
            self.box3_2['values'] = tuple(dColumns)
            
        self.box3_1 = ttk.Combobox(self.lb3,width=20,textvariable=self.groupname,postcommand=upd_combobox_g)
        self.box3_1.grid(row=1,column=0)
        #self.box3_1.current(1)
        
        self.box3_2 = ttk.Combobox(self.lb3,width=20,textvariable=self.gradename,postcommand=upd_combobox_c)
        dfCloumns = self.dataframe.columns
        print (dfCloumns)
        self.box3_2['values'] = tuple(dfCloumns)
        self.box3_2.grid(row=1,column=1)
        #self.box3_2.current(1)
        
        def addg(df,group,grade,path):
            try:
                df.set_index(['Group'],inplace=True)
            except KeyError:
                pass
            df.loc[group,grade] += 1
            df.reset_index(inplace=True)
            self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
            self.dataframe = df
            pt.show()
            df.to_csv(path, index=False)
            
        def reduceg(df,group,grade,path):
            try:
                df.set_index(['Group'],inplace=True)
            except KeyError:
                pass
            df.loc[group,grade] -= 1
            df.reset_index(inplace=True)
            self.table = pt = Table(self.lb1,dataframe=df,\
                                            showtoolbar=True, showstatusbar=True)
            self.dataframe = df
            pt.show()
            df.to_csv(path, index=False)
        
        self.button3_1 = Button(self.lb3,text='ADD',command=lambda: addg(self.dataframe,self.box3_1.get(),self.box3_2.get(),self.fentry.get()))
        self.button3_1.grid(row=2,column=0)
        
        self.button3_2 = Button(self.lb3,text='SUBTRACT',command=lambda: reduceg(self.dataframe,self.box3_1.get(),self.box3_2.get(),self.fentry.get()))
        self.button3_2.grid(row=2,column=1)
        
        
        self.lb4 = LabelFrame(object,width=w, height=h*0.1, text='File',padx=8)
        self.lb4.grid(row=5,column=0,columnspan=6,padx=2,pady=2)
        self.flabel = Label(self.lb4,text='ChooseFilePath:')
        self.flabel.grid(row=0,column=0)
        
        
        def fileopen():
            filename_ = tf.askopenfilename()
            self.filename.set(filename_)
            print (filename_)
            print (self.filename)
            
        def refresh(path):
            if not path:
                print ('Nothing happen!')
            else:
                print (path)
                self.dataframe = read_csv(path,encoding='gbk')
                self.table = pt = Table(self.lb1,dataframe=self.dataframe,\
                                        showtoolbar=True, showstatusbar=True)
                pt.show()
                print (self.dataframe.head())
                return self.dataframe
                
        def savefile(df,path):
             print (df.info())
             df.iloc[:,-1] = 0
             for i in range(2,df.shape[1]):
                 df.iloc[:,-1] += df.iloc[:,-i].astype(float)
             self.dataframe = df
             self.table = pt = Table(self.lb1,dataframe=self.dataframe,\
                                        showtoolbar=True, showstatusbar=True)
             pt.show()
             df.to_csv(path, index=False)
             print (df.head(3))
        
        self.fentry = Entry(self.lb4,textvariable=self.filename)
        self.fentry.grid(row=0,column=1)
        
        self.fbutton1 = Button(self.lb4,text='...',command=fileopen)
        self.fbutton1.grid(row=0,column=2)
        
        self.fbutton2 = Button(self.lb4,text='ReadFile',command=lambda: refresh(self.fentry.get()))
        self.fbutton2.grid(row=0,column=3)
        
        self.fbutton3 = Button(self.lb4,text='SaveFile',command=lambda: savefile(self.dataframe,self.fentry.get()))
        self.fbutton3.grid(row=0,column=4)
        
        
        
        def createPage():
            pass
        
        self.button = Button(object,text='SimpleModel',command=createPage)
        self.button.grid(row=5, column=6)
        

root = Tk()
root.title("CalcuGradeSoftware")
app = App(root)
root.mainloop()
