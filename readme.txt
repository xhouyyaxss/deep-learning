git init 初始化，为当前仓库创建仓库进行跟踪，版本控制
git config --gloabal  user.name
git config --gloabal  user.email

git add filename 将文件添加到缓存区stage(没添加缓存区是红色的，添加时绿色的)
git status 查看当前是否还有没有提交的分支
git commit -m “xxxx”  提交到工作区-m后跟说明
git log 查看历史记录 只能查看当前版本的之前的版本
git log --pretty=oneline 也是查看历史记录，缩略信息
git reset --hard head^ 回退到上一个版本
git reset --hard head^^ 上上版本
git  reset --hard head~100 回退上100版本
git reflog 可以查看所有的版本号
git checkout -- file可以丢弃工作区的修改 file名字前有空格
新建分支
git pull origin master 从远程拉取代码并与当前分支合并
git branch <分支名> 新建分支
git checkout <分支名> 切换到分支
git push origin <分支名> 把本地的分支推送到远程，让远端也有一个你的分支，用来后面提交代码
git checkout -b <分支名>创建分支名并且切换到该分支
git pull --rebase origin 远程分支名 拿到罪行的远程分支
git pull -rebase origin <分支名>
git pull -rebase origin master
(解决冲突之后：git add git rebase -continue)
git branch -d <分支名> 删除本地分支
git push origin --delete <分支名>删除远端分支
git branch 查看本地有哪些分支
git checkout <分支名>
如何让从远程拉一个分支，如果是更新就用pull
git fetch origin <分支名>: <分支名>
合并分支
基于master创建了一个dev分支，基于dev创建了一个test分支，如果要合并，先git checkout dev，再git merge test(在dev分支上将test分支合并到dev上)
githubtoken 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
这是新的txt

