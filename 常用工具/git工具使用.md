# Git 工具使用详解

Git 是一个分布式版本控制系统，用于高效地处理从小型到大型项目的所有内容。

---

## 一、Git 基础配置

### 1.1 配置用户信息
```bash
# 配置用户名
git config --global user.name "Your Name"

# 配置邮箱
git config --global user.email "your.email@example.com"

# 查看配置
git config --list

# 查看特定配置
git config user.name
```

### 1.2 配置编辑器
```bash
# 设置默认编辑器为 vim
git config --global core.editor vim

# 设置默认编辑器为 nano
git config --global core.editor nano

# 设置默认编辑器为 VS Code
git config --global core.editor "code --wait"
```

### 1.3 配置别名（Aliases）
```bash
# 设置快捷命令
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

---

## 二、Git 仓库操作

### 2.1 创建仓库
```bash
# 在当前目录初始化新仓库
git init

# 克隆远程仓库
git clone <repository-url>

# 克隆指定分支
git clone -b <branch-name> <repository-url>

# 克隆到指定目录
git clone <repository-url> <directory-name>

# 克隆时只获取最新版本（浅克隆）
git clone --depth 1 <repository-url>
```

### 2.2 查看仓库状态
```bash
# 查看工作区状态
git status

# 查看简短状态
git status -s

# 查看详细状态
git status -v
```

---

## 三、文件操作

### 3.1 添加文件到暂存区
```bash
# 添加单个文件
git add <file-name>

# 添加所有修改的文件
git add .

# 添加所有修改和删除的文件
git add -A

# 交互式添加
git add -i

# 添加时忽略某些文件
git add --all -- ':!path/to/ignore'
```

### 3.2 提交更改
```bash
# 提交暂存区的更改
git commit -m "提交信息"

# 提交并添加所有修改过的文件（不包括新文件）
git commit -am "提交信息"

# 修改最后一次提交
git commit --amend -m "新的提交信息"

# 修改最后一次提交但不修改信息
git commit --amend --no-edit
```

### 3.3 撤销操作
```bash
# 撤销工作区的修改（未暂存）
git checkout -- <file-name>

# 撤销暂存区的修改（保留工作区修改）
git reset HEAD <file-name>

# 撤销暂存区和工作区的所有修改（危险操作）
git reset --hard HEAD

# 撤销最后一次提交（保留修改）
git reset --soft HEAD~1

# 撤销最后一次提交（不保留修改）
git reset --hard HEAD~1
```

### 3.4 删除和重命名
```bash
# 删除文件
git rm <file-name>

# 强制删除
git rm -f <file-name>

# 从暂存区删除但保留文件
git rm --cached <file-name>

# 重命名文件
git mv <old-name> <new-name>
```

---

## 四、分支管理

### 4.1 查看分支
```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r

# 查看所有分支
git branch -a

# 查看分支详细信息
git branch -vv

# 查看已合并的分支
git branch --merged

# 查看未合并的分支
git branch --no-merged
```

### 4.2 创建和切换分支
```bash
# 创建新分支
git branch <branch-name>

# 创建并切换到新分支
git checkout -b <branch-name>
# 或
git switch -c <branch-name>

# 切换到已有分支
git checkout <branch-name>
# 或
git switch <branch-name>

# 基于远程分支创建本地分支
git checkout -b <branch-name> origin/<branch-name>
```

### 4.3 合并分支
```bash
# 合并指定分支到当前分支
git merge <branch-name>

# 合并时禁止快进
git merge --no-ff <branch-name>

# 合并时只允许快进
git merge --ff-only <branch-name>

# 取消合并
git merge --abort
```

### 4.4 删除分支
```bash
# 删除已合并的本地分支
git branch -d <branch-name>

# 强制删除本地分支
git branch -D <branch-name>

# 删除远程分支
git push origin --delete <branch-name>
# 或
git push origin :<branch-name>
```

### 4.5 储藏（Stash）
```bash
# 储藏当前修改
git stash

# 储藏并添加描述
git stash save "描述信息"

# 查看储藏列表
git stash list

# 应用最近一次储藏
git stash apply

# 应用指定储藏
git stash apply stash@{n}

# 应用并删除最近一次储藏
git stash pop

# 删除最近一次储藏
git stash drop

# 删除所有储藏
git stash clear

# 查看储藏内容
git stash show -p stash@{n}
```

---

## 五、远程仓库操作

### 5.1 查看远程仓库
```bash
# 查看远程仓库
git remote -v

# 查看远程仓库详细信息
git remote show origin

# 添加远程仓库
git remote add origin <repository-url>

# 修改远程仓库地址
git remote set-url origin <new-url>

# 删除远程仓库
git remote remove origin
```

### 5.2 推送和拉取
```bash
# 推送到远程仓库
git push origin <branch-name>

# 首次推送并关联分支
git push -u origin <branch-name>

# 强制推送（谨慎使用）
git push -f origin <branch-name>

# 拉取远程更新
git pull origin <branch-name>

# 拉取但不合并
git fetch origin

# 拉取所有分支
git fetch --all

# 拉取并变基
git pull --rebase origin <branch-name>
```

### 5.3 同步远程分支
```bash
# 获取远程分支列表
git fetch origin

# 创建本地分支跟踪远程分支
git checkout --track origin/<branch-name>

# 设置已有本地分支跟踪远程分支
git branch -u origin/<branch-name>

# 清理已删除的远程分支引用
git remote prune origin
```

---

## 六、查看历史记录

### 6.1 查看提交历史
```bash
# 查看提交历史
git log

# 查看简洁历史
git log --oneline

# 查看图形化历史
git log --graph --oneline --all

# 查看最近 n 条提交
git log -n 5

# 查看文件修改历史
git log --follow -- <file-name>

# 查看每次提交的统计信息
git log --stat

# 查看每次提交的详细修改
git log -p

# 按作者筛选
git log --author="用户名"

# 按日期筛选
git log --since="2024-01-01" --until="2024-12-31"

# 按提交信息筛选
git log --grep="关键词"
```

### 6.2 查看具体提交
```bash
# 查看某次提交的详细信息
git show <commit-hash>

# 查看某次提交中某个文件的修改
git show <commit-hash>:<file-name>

# 查看某文件的历史修改
git blame <file-name>

# 查看工作区与暂存区的差异
git diff

# 查看暂存区与最新提交的差异
git diff --cached
# 或
git diff --staged

# 查看工作区与最新提交的差异
git diff HEAD

# 查看两次提交之间的差异
git diff <commit1> <commit2>

# 查看某文件在两个版本间的差异
git diff <commit1> <commit2> -- <file-name>
```

---

## 七、标签管理

### 7.1 创建标签
```bash
# 创建轻量标签
git tag <tag-name>

# 创建附注标签
git tag -a <tag-name> -m "标签说明"

# 为历史提交创建标签
git tag -a <tag-name> <commit-hash> -m "标签说明"
```

### 7.2 查看和推送标签
```bash
# 查看所有标签
git tag

# 查看标签详细信息
git show <tag-name>

# 推送单个标签到远程
git push origin <tag-name>

# 推送所有标签到远程
git push origin --tags

# 删除本地标签
git tag -d <tag-name>

# 删除远程标签
git push origin --delete tag <tag-name>
```

---

## 八、变基（Rebase）

### 8.1 基本变基
```bash
# 将当前分支变基到指定分支
git rebase <branch-name>

# 交互式变基（修改历史）
git rebase -i HEAD~n

# 继续变基（解决冲突后）
git rebase --continue

# 跳过当前提交
git rebase --skip

# 取消变基
git rebase --abort
```

### 8.2 常用交互式变基操作
```
pick    # 保留提交
reword  # 修改提交信息
edit    # 修改提交内容
squash  # 合并到上一个提交
fixup   # 合并到上一个提交，丢弃提交信息
drop    # 删除提交
```

---

## 九、子模块（Submodule）

### 9.1 子模块操作
```bash
# 添加子模块
git submodule add <repository-url> <path>

# 初始化子模块
git submodule init

# 更新子模块
git submodule update

# 初始化并更新子模块
git submodule update --init --recursive

# 克隆包含子模块的仓库
git clone --recursive <repository-url>

# 删除子模块
# 1. 删除 .gitmodules 中相关配置
# 2. 删除 .git/config 中相关配置
# 3. 执行 git rm --cached <path>
# 4. 删除文件夹 rm -rf <path>
```

---

## 十、Cherry-pick

```bash
# 将指定提交应用到当前分支
git cherry-pick <commit-hash>

# 应用多个提交
git cherry-pick <commit1> <commit2>

# 应用提交范围
git cherry-pick <commit1>^..<commit2>

# 继续 cherry-pick（解决冲突后）
git cherry-pick --continue

# 取消 cherry-pick
git cherry-pick --abort
```

---

## 十一、Git 工作流

### 11.1 常见工作流

**功能分支工作流（Feature Branch Workflow）**
```bash
# 1. 从主分支创建功能分支
git checkout -b feature/login main

# 2. 开发功能并提交
git add .
git commit -m "实现登录功能"

# 3. 推送到远程
git push origin feature/login

# 4. 创建 Pull Request 进行代码审查

# 5. 审查通过后合并到主分支
git checkout main
git merge feature/login

# 6. 删除功能分支
git branch -d feature/login
```

**Git Flow 工作流**
```bash
# 初始化 Git Flow
git flow init

# 创建功能分支
git flow feature start login

# 完成功能分支
git flow feature finish login

# 创建发布分支
git flow release start 1.0.0

# 完成发布分支
git flow release finish 1.0.0

# 创建热修复分支
git flow hotfix start fix-bug

# 完成热修复分支
git flow hotfix finish fix-bug
```

---

## 十二、常见问题解决

### 12.1 解决合并冲突
```bash
# 1. 当合并出现冲突时，Git 会标记冲突文件
# 2. 手动编辑冲突文件，解决冲突
# 3. 添加解决后的文件
git add <resolved-file>

# 4. 完成合并
git commit -m "解决合并冲突"
# 或如果是 rebase
git rebase --continue
```

### 12.2 撤销已推送的提交
```bash
# 方法1：使用 revert（推荐，不会重写历史）
git revert <commit-hash>
git push origin <branch-name>

# 方法2：使用 reset（会重写历史，谨慎使用）
git reset --hard <commit-hash>
git push -f origin <branch-name>
```

### 12.3 找回丢失的提交
```bash
# 查看引用日志
git reflog

# 恢复到指定状态
git reset --hard <reflog-entry>

# 查看所有引用的历史
git reflog show --all
```

### 12.4 大文件处理
```bash
# 查看仓库中最大的文件
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort -rnk3 | head -20

# 使用 Git LFS 管理大文件
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.psd"
git lfs track "*.zip"

# 查看追踪的文件类型
git lfs track
```

---

## 十三、Git 配置最佳实践

### 13.1 推荐的全局配置
```bash
# 设置默认分支名为 main
git config --global init.defaultBranch main

# 设置 pull 策略为 rebase
git config --global pull.rebase true

# 设置 push 策略为 simple
git config --global push.default simple

# 自动设置远程分支跟踪
git config --global push.autoSetupRemote true

# 设置颜色输出
git config --global color.ui auto

# 设置换行符自动转换（Windows）
git config --global core.autocrlf true

# 设置换行符不转换（Linux/Mac）
git config --global core.autocrlf input

# 设置忽略文件权限变化
git config --global core.fileMode false

# 设置 Git 缓存密码（15分钟）
git config --global credential.helper cache

# 永久缓存密码
git config --global credential.helper store
```

### 13.2 实用的 Git 别名
```bash
# 日志别名
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

git config --global alias.lga "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all"

# 状态别名
git config --global alias.st "status -sb"

# 提交别名
git config --global alias.cm "commit -m"
git config --global alias.cam "commit -am"
git config --global alias.cane "commit --amend --no-edit"

# 分支别名
git config --global alias.br "branch -vv"
git config --global alias.bra "branch -a -vv"

# 差异别名
git config --global alias.diffs "diff --staged"
git config --global alias.diffc "diff --cached"

# 其他实用别名
git config --global alias.unstage "reset HEAD --"
git config --global alias.last "log -1 HEAD --stat"
git config --global alias.visual "!gitk"
```

---

## 十四、.gitignore 文件

### 14.1 常用 .gitignore 规则
```gitignore
# 忽略所有 .log 文件
*.log

# 忽略所有 .tmp 文件
*.tmp

# 忽略 node_modules 目录
node_modules/

# 忽略编译输出
dist/
build/
target/

# 忽略 IDE 配置文件
.idea/
.vscode/
*.iml

# 忽略操作系统文件
.DS_Store
Thumbs.db

# 忽略环境配置文件
.env
.env.local

# 不忽略特定的 .log 文件
!important.log

# 忽略某个目录下的所有文件，但不忽略目录本身
logs/*
!logs/.gitkeep
```

### 14.2 全局 .gitignore
```bash
# 设置全局 .gitignore
git config --global core.excludesfile ~/.gitignore_global

# 创建全局 .gitignore 文件
touch ~/.gitignore_global
```

---

## 十五、Git 钩子和自动化

### 15.1 常用 Git 钩子
Git 钩子存储在 `.git/hooks/` 目录下：

- `pre-commit`：提交前执行
- `prepare-commit-msg`：提交信息编辑器打开前执行
- `commit-msg`：提交信息编辑完成后执行
- `post-commit`：提交完成后执行
- `pre-push`：推送前执行
- `post-checkout`：切换分支后执行
- `post-merge`：合并完成后执行

### 15.2 示例：pre-commit 钩子
```bash
#!/bin/bash
# .git/hooks/pre-commit

# 检查是否有未解决的合并冲突标记
if grep -r "<<<<<<< HEAD" . --include="*.py" --include="*.js" --include="*.java" --include="*.md"; then
    echo "错误：发现未解决的合并冲突标记"
    exit 1
fi

# 运行代码格式化检查
if command -v black &> /dev/null; then
    black --check .
    if [ $? -ne 0 ]; then
        echo "错误：代码格式化检查失败，请运行 'black .' 格式化代码"
        exit 1
    fi
fi

exit 0
```

---

## 十六、Git 与其他工具集成

### 16.1 与 SSH 集成
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your.email@example.com"

# 添加 SSH 密钥到 ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 复制公钥到剪贴板（Linux）
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard

# 测试 SSH 连接
git@github.com
```

### 16.2 与 GPG 集成（签名提交）
```bash
# 生成 GPG 密钥
gpg --full-generate-key

# 列出 GPG 密钥
gpg --list-secret-keys --keyid-format=long

# 配置 Git 使用 GPG 签名
git config --global user.signingkey <GPG-key-id>
git config --global commit.gpgsign true

# 创建签名提交
git commit -S -m "签名提交"

# 验证提交签名
git verify-commit <commit-hash>
```

---

## 十七、Git 性能优化

### 17.1 大型仓库优化
```bash
# 启用部分克隆（Git 2.25+）
git clone --filter=blob:none <repository-url>

# 启用稀疏检出
git sparse-checkout init --cone
git sparse-checkout set <directory1> <directory2>

# 清理和优化仓库
git gc

# 深度清理
git gc --aggressive --prune=now

# 查看仓库大小
git count-objects -vH
```

### 17.2 加速操作
```bash
# 启用文件系统监视器（Git 2.31+）
git config --global core.fsmonitor true

# 启用.untrackedCache
git config --global core.untrackedCache true

# 并行获取子模块
git config --global submodule.fetchJobs 8
```

---

## 十八、Git 命令速查表

| 操作 | 命令 |
|------|------|
| 初始化仓库 | `git init` |
| 克隆仓库 | `git clone <url>` |
| 查看状态 | `git status` |
| 添加文件 | `git add <file>` |
| 提交更改 | `git commit -m "msg"` |
| 推送代码 | `git push origin <branch>` |
| 拉取代码 | `git pull origin <branch>` |
| 创建分支 | `git branch <name>` |
| 切换分支 | `git checkout <branch>` |
| 合并分支 | `git merge <branch>` |
| 查看日志 | `git log --oneline` |
| 查看差异 | `git diff` |
| 储藏修改 | `git stash` |
| 恢复储藏 | `git stash pop` |
| 创建标签 | `git tag -a <name> -m "msg"` |
| 变基 | `git rebase <branch>` |

---

> **提示**：Git 功能非常强大，建议在日常使用中多实践。遇到问题时，`git status` 和 `git log` 是最好的诊断工具。
