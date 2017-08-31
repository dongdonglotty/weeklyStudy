> git 在网上的使用教程都快烂掉了，我感觉我也写不出花来了，所以接下来我写的东西属于入门以后的一些东西，
当然啦，我是现学现卖啦。

## git config  相关

#### 说明
  就这个命令来说，通常的教程上都是教你这样配置的：  

```bash
# --global 表示当前用户级别的配置文件，在linux下是 ~/.gitconfig 或者 ~/.config/git/config
$ git config --global -l  
$ git config --global user.name "####"
$ git config --global user.email "#####"


# --system 表示系统级别的配置文件，在linux比如说是 /etc/gitconfig 等等
$ git config --system -l
# 略

# --local 表示仓库级别的配置文件
$ git config --local -l
# 略
```

#### 练习
> 习题： 请在当前系统上一次输入上述三个命令，查看结果


#### 用处

  个人觉得一般情况下，不用关心这个，除非在公司里面，你想上班用一个账号，下班写自己的代码用一个账号，这样的话，才需要你像上面那样  
配置。当然，这个问题有顺带提到了ssh免密码输入的问题，原理是这样子的：使用ssh生成一个公钥和密钥，然后将公钥存在github上，这样双方在进行验证后就可以直接通信了。
  
  如果有两个账号的话，比如说一个是个人的，一个是公司的怎么办呢？这个时候需要在 `~/.ssh/config` 文件中稍加配置了：
```conf
Host github.com
  User git
  IdentityFile ~/.ssh/personal

Host company.com
    User git
    IdentityFile ~/.ssh/company
    
```

这就是在这个命令中可能会使用到的一些情况。


## git 常用的几个命令

#### 说明

```bash
# 不加 --staged 表示可以显示当前正在修改的文件与上一版本的文件的区别
# 加 --staged 表示可以比较已经staged了的文件与上一版本文件的区别
$ git diff [--staged]

#   这个命令是用来unstaged文件的，一般情况下，git add [file] 以后，
#文件的状态就变成 staged，如果不想加入这个文件到本地仓库，可以使用这个命令
$ git reset [file]

# 这个命令比较复杂，主要是用来重置整个提交的commit的（没错就是每次提交的唯一标识）
# --hard HEAD 比较常用，意思是将当前做出的所有的修改消除掉，重置索引和当前工作的分支
# --mixed     默认的选项，意思是重置当前的索引，但是并不重置当前的工作分支，简单来说就是不tracked文件
$ git reset [mode] [commit] 


# 这个命令最常用
$ git commit -m [message]

# 列出当前所有的分支
# 创建一个新的分支
# 删除一个分支
$ git branch [-l|--list]
$ git branch [branch-name]
$ git branch [-d|--delete]

```