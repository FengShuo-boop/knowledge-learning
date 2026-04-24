# Tmux 工具使用详解

Tmux（Terminal Multiplexer）是一个终端复用器，允许你在单个终端窗口中创建多个会话、窗口和面板，极大地提高了终端工作效率。

---

## 一、Tmux 简介

### 1.1 什么是 Tmux
Tmux 是一个开源的终端复用工具，它可以：
- **会话管理**：保持会话在后台运行，即使断开 SSH 连接也不会中断任务
- **窗口管理**：在一个终端中创建多个窗口（类似浏览器的标签页）
- **面板分割**：将窗口分割成多个面板，同时查看和操作多个终端
- **共享会话**：允许多个用户同时连接到同一个会话

### 1.2 Tmux 的优势
- **持久性**：网络断开不会丢失工作进度
- **多任务**：同时运行和监控多个程序
- **效率**：无需频繁切换终端窗口
- **协作**：可以共享终端会话进行协作

### 1.3 安装 Tmux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tmux

# CentOS/RHEL/Fedora
sudo yum install tmux
# 或
sudo dnf install tmux

# macOS
brew install tmux

# Arch Linux
sudo pacman -S tmux

# 验证安装
tmux -V
```

---

## 二、基础概念

Tmux 采用三层结构组织：

```
Session（会话）
├── Window（窗口）1
│   ├── Pane（面板）1
│   ├── Pane（面板）2
│   └── Pane（面板）3
├── Window（窗口）2
│   ├── Pane（面板）1
│   └── Pane（面板）2
└── Window（窗口）3
    └── Pane（面板）1
```

### 2.1 会话（Session）
- 最高级别的组织单位
- 可以包含多个窗口
- 可以在后台保持运行
- 可以随时附加（attach）和分离（detach）

### 2.2 窗口（Window）
- 类似终端的标签页
- 一个会话可以包含多个窗口
- 每次只能显示一个窗口
- 窗口可以重命名和重新排序

### 2.3 面板（Pane）
- 窗口的分割区域
- 一个窗口可以分割成多个面板
- 每个面板可以独立运行命令
- 面板可以同时显示

### 2.4 前缀键（Prefix Key）
Tmux 的所有快捷键都需要先按**前缀键**，默认是 `Ctrl+b`（简写为 `C-b`）。

> **提示**：你可以将前缀键修改为更容易按的组合，如 `Ctrl+a`。

---

## 三、会话管理

### 3.1 创建会话
```bash
# 创建新会话
tmux

# 创建命名会话（推荐）
tmux new -s mysession
# 或
tmux new-session -s mysession

# 创建会话并运行命令
tmux new -s mysession -d 'top'

# 创建会话时指定窗口名称
tmux new -s mysession -n window1
```

### 3.2 分离会话
```bash
# 在 Tmux 内部按：
Ctrl+b d

# 或输入命令
tmux detach
```

### 3.3 列出会话
```bash
# 列出所有会话
tmux ls
# 或
tmux list-sessions

# 输出示例：
# mysession: 2 windows (created Sat Jan 15 10:30:00 2024)
# work: 3 windows (created Sat Jan 15 11:00:00 2024) [attached]
```

### 3.4 附加会话
```bash
# 附加到指定会话
tmux attach -t mysession
# 或
tmux attach-session -t mysession

# 简写形式
tmux a -t mysession

# 附加到最后一个使用的会话
tmux attach
# 或
tmux a
```

### 3.5 切换会话
在 Tmux 内部：
```bash
# 切换到上一个会话
Ctrl+b (

# 切换到下一个会话
Ctrl+b )

# 交互式选择会话
Ctrl+b s

# 快速切换到指定会话（需要配置）
Ctrl+b L  # 切换到最后一个会话
```

### 3.6 重命名会话
```bash
# 在 Tmux 内部
Ctrl+b $

# 或命令行
tmux rename-session -t oldname newname
```

### 3.7 关闭和删除会话
```bash
# 在 Tmux 内部输入 exit 或按 Ctrl+d

# 命令行关闭指定会话
tmux kill-session -t mysession

# 关闭所有会话（除了当前）
tmux kill-session -a

# 关闭所有会话（包括当前）
tmux kill-server
```

---

## 四、窗口管理

### 4.1 创建窗口
```bash
# 在 Tmux 内部创建新窗口
Ctrl+b c

# 创建窗口并命名
Ctrl+b c  # 然后重命名

# 命令行创建窗口
tmux new-window -t mysession -n windowname
```

### 4.2 切换窗口
```bash
# 切换到下一个窗口
Ctrl+b n

# 切换到上一个窗口
Ctrl+b p

# 切换到指定编号的窗口
Ctrl+b 0    # 切换到窗口 0
Ctrl+b 1    # 切换到窗口 1
Ctrl+b 2    # 切换到窗口 2
...
Ctrl+b 9    # 切换到窗口 9

# 交互式选择窗口
Ctrl+b w

# 切换到上一个活动的窗口
Ctrl+b l

# 按名称查找窗口
Ctrl+b f
```

### 4.3 重命名窗口
```bash
# 在 Tmux 内部
Ctrl+b ,

# 然后输入新名称，按 Enter 确认

# 命令行重命名
tmux rename-window -t mysession:0 newname
```

### 4.4 关闭窗口
```bash
# 在窗口中输入 exit 或按 Ctrl+d

# 强制关闭窗口
Ctrl+b &
# 然后按 y 确认

# 命令行关闭
tmux kill-window -t mysession:0
```

### 4.5 窗口布局
```bash
# 窗口列表显示在底部状态栏
# 当前窗口用 * 标记，上一个窗口用 - 标记

# 交换窗口位置
Ctrl+b :swap-window -s 0 -t 1

# 移动窗口到指定位置
Ctrl+b :move-window -t 2
```

---

## 五、面板管理

### 5.1 分割面板
```bash
# 垂直分割（左右）
Ctrl+b %

# 水平分割（上下）
Ctrl+b "

# 使用当前面板路径分割（Tmux 1.9+）
# 需要先配置，见配置文件部分

# 命令行分割
tmux split-window -h    # 水平分割
tmux split-window -v    # 垂直分割
tmux split-window -h -c "#{pane_current_path}"  # 保持当前目录
```

### 5.2 切换面板
```bash
# 使用方向键切换
Ctrl+b ←    # 左
Ctrl+b →    # 右
Ctrl+b ↑    # 上
Ctrl+b ↓    # 下

# 循环切换面板
Ctrl+b o    # 切换到下一个面板
Ctrl+b ;    # 切换到最后一个活动的面板

# 按编号切换
Ctrl+b q    # 显示面板编号，然后按数字键切换

# 交互式选择
Ctrl+b q    # 显示编号后按对应数字
```

### 5.3 调整面板大小
```bash
# 使用方向键调整（按住 Ctrl+b 不放，再按方向键）
Ctrl+b Ctrl+←     # 向左缩小
trl+b Ctrl+→     # 向右扩大
Ctrl+b Ctrl+↑     # 向上缩小
Ctrl+b Ctrl+↓     # 向下扩大

# 使用预设布局
Ctrl+b Alt+1      # 水平等分布局
Ctrl+b Alt+2      # 垂直等分布局
Ctrl+b Alt+3      # 主面板水平布局
Ctrl+b Alt+4      # 主面板垂直布局
Ctrl+b Alt+5      # 平铺布局
Ctrl+b Space      # 循环切换布局

# 最大化/恢复面板
Ctrl+b z          # 最大化当前面板，再次按恢复
```

### 5.4 关闭面板
```bash
# 在面板中输入 exit 或按 Ctrl+d

# 强制关闭当前面板
Ctrl+b x
# 然后按 y 确认

# 关闭其他所有面板（保留当前）
Ctrl+b !
```

### 5.5 移动面板
```bash
# 将面板移动到另一个窗口
Ctrl+b :join-pane -t :1

# 将面板拆分为新窗口
Ctrl+b !

# 交换面板位置
Ctrl+b Ctrl+o     # 顺时针旋转面板
Ctrl+b Alt+o      # 逆时针旋转面板
Ctrl+b {          # 与上一个面板交换
Ctrl+b }          # 与下一个面板交换
```

---

## 六、复制模式

### 6.1 进入复制模式
```bash
# 进入复制模式（使用 vi 风格按键）
Ctrl+b [

# 进入复制模式并滚动到上一页
Ctrl+b PageUp
```

### 6.2 复制模式中的导航
```bash
# 基本移动
h       # 左
j       # 下
k       # 上
l       # 右

# 快速移动
w       # 下一个单词开头
b       # 上一个单词开头
e       # 单词结尾

# 行内移动
0       # 行首
$       # 行尾
^       # 第一个非空字符

# 翻页
Ctrl+u  # 上半页
Ctrl+d  # 下半页
Ctrl+b  # 上一页
Ctrl+f  # 下一页

# 文件移动
g       # 文件开头
G       # 文件结尾

# 搜索
/       # 向下搜索
?       # 向上搜索
n       # 下一个匹配
N       # 上一个匹配
```

### 6.3 选择和复制文本
```bash
# 开始选择
Space   # 开始选择（vi 模式）
v       # 字符选择
V       # 行选择

# 复制
Enter   # 复制并退出复制模式
y       # 复制（vi 模式）

# 取消
q       # 退出复制模式
Esc     # 退出复制模式

# 粘贴
Ctrl+b ]    # 粘贴缓冲区内容
```

### 6.4 缓冲区管理
```bash
# 查看缓冲区列表
Ctrl+b #

# 选择并粘贴缓冲区
Ctrl+b =

# 命令行操作缓冲区
tmux list-buffers       # 列出缓冲区
tmux show-buffer        # 显示当前缓冲区内容
tmux save-buffer file   # 保存缓冲区到文件
tmux delete-buffer      # 删除当前缓冲区
```

---

## 七、常用快捷键汇总

### 7.1 会话快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b d` | 分离当前会话 |
| `Ctrl+b D` | 选择并分离会话 |
| `Ctrl+b s` | 列出并切换会话 |
| `Ctrl+b $` | 重命名当前会话 |
| `Ctrl+b (` | 切换到上一个会话 |
| `Ctrl+b )` | 切换到下一个会话 |
| `Ctrl+b L` | 切换到最后一个活动的会话 |

### 7.2 窗口快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b c` | 创建新窗口 |
| `Ctrl+b &` | 关闭当前窗口 |
| `Ctrl+b 0-9` | 切换到指定窗口 |
| `Ctrl+b p` | 切换到上一个窗口 |
| `Ctrl+b n` | 切换到下一个窗口 |
| `Ctrl+b l` | 切换到最后一个活动的窗口 |
| `Ctrl+b w` | 列出并选择窗口 |
| `Ctrl+b ,` | 重命名当前窗口 |
| `Ctrl+b f` | 按名称查找窗口 |
| `Ctrl+b .` | 移动窗口到指定编号 |

### 7.3 面板快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b %` | 垂直分割面板 |
| `Ctrl+b "` | 水平分割面板 |
| `Ctrl+b x` | 关闭当前面板 |
| `Ctrl+b z` | 最大化/恢复面板 |
| `Ctrl+b !` | 将面板拆分为新窗口 |
| `Ctrl+b o` | 切换到下一个面板 |
| `Ctrl+b ;` | 切换到最后一个活动的面板 |
| `Ctrl+b q` | 显示面板编号 |
| `Ctrl+b {` | 与上一个面板交换 |
| `Ctrl+b }` | 与下一个面板交换 |
| `Ctrl+b Space` | 循环切换面板布局 |
| `Ctrl+b Alt+1-5` | 应用预设布局 |
| `Ctrl+b Ctrl+方向键` | 调整面板大小 |

### 7.4 其他快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b ?` | 显示所有快捷键帮助 |
| `Ctrl+b :` | 进入命令模式 |
| `Ctrl+b [` | 进入复制模式 |
| `Ctrl+b ]` | 粘贴缓冲区内容 |
| `Ctrl+b =` | 选择并粘贴缓冲区 |
| `Ctrl+b t` | 显示时钟 |
| `Ctrl+b r` | 重新加载配置文件 |
| `Ctrl+b ~` | 显示 Tmux 日志 |

---

## 八、命令模式

### 8.1 进入命令模式
```bash
# 在 Tmux 内部
Ctrl+b :

# 然后输入命令，按 Enter 执行
```

### 8.2 常用命令
```bash
# 会话管理
new-session -s name           # 创建会话
rename-session -t old new     # 重命名会话
kill-session -t name          # 关闭会话

# 窗口管理
new-window -n name            # 创建窗口
rename-window -t 0 newname    # 重命名窗口
kill-window -t 0              # 关闭窗口
swap-window -s 0 -t 1         # 交换窗口

# 面板管理
split-window -h               # 水平分割
split-window -v               # 垂直分割
kill-pane                     # 关闭面板
resize-pane -L 10             # 向左调整 10 格
resize-pane -R 10             # 向右调整 10 格
resize-pane -U 5              # 向上调整 5 格
resize-pane -D 5              # 向下调整 5 格
select-pane -t 0              # 选择面板
swap-pane -s 0 -t 1           # 交换面板

# 其他
source-file ~/.tmux.conf      # 重新加载配置
set-option key value          # 设置选项
show-options                  # 显示选项
```

---

## 九、配置文件

### 9.1 配置文件位置
```bash
# 主配置文件
~/.tmux.conf

# 系统配置文件
/etc/tmux.conf
```

### 9.2 推荐配置
```bash
# ~/.tmux.conf

# =============================================
# 基本设置
# =============================================

# 设置前缀键为 Ctrl+a（更容易按）
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# 启用鼠标支持
set -g mouse on

# 设置默认终端
set -g default-terminal "screen-256color"

# 设置历史缓冲区大小
set -g history-limit 10000

# 设置状态栏更新间隔
set -g status-interval 1

# 设置窗口索引从 1 开始
set -g base-index 1
setw -g pane-base-index 1

# 重新编号窗口（关闭后自动重新编号）
set -g renumber-windows on

# 启用 vi 模式
setw -g mode-keys vi

# 设置状态栏位置
set -g status-position bottom

# =============================================
# 快捷键绑定
# =============================================

# 重新加载配置文件
bind r source-file ~/.tmux.conf \; display "配置已重新加载！"

# 使用 | 和 - 分割面板（更直观）
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# 使用 vim 风格切换面板
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# 使用 Alt+方向键快速切换面板（无需前缀键）
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# 使用 Shift+方向键调整面板大小
bind -n S-Left resize-pane -L 2
bind -n S-Right resize-pane -R 2
bind -n S-Up resize-pane -U 2
bind -n S-Down resize-pane -D 2

# 使用 Alt+数字快速切换窗口（无需前缀键）
bind -n M-1 select-window -t 1
bind -n M-2 select-window -t 2
bind -n M-3 select-window -t 3
bind -n M-4 select-window -t 4
bind -n M-5 select-window -t 5

# 快速创建新窗口
bind c new-window -c "#{pane_current_path}"

# 快速关闭面板（无需确认）
bind x kill-pane

# 复制模式使用 vi 风格
bind-key -T copy-mode-vi v send-keys -X begin-selection
bind-key -T copy-mode-vi y send-keys -X copy-selection-and-cancel
bind-key -T copy-mode-vi r send-keys -X rectangle-toggle

# 粘贴
bind p paste-buffer

# =============================================
# 外观设置
# =============================================

# 状态栏样式
set -g status-style bg=black,fg=white
set -g status-left-length 40
set -g status-right-length 100

# 状态栏左侧：会话名称
set -g status-left "#[fg=green]Session: #S #[fg=yellow]#I #[fg=cyan]#P"

# 状态栏右侧：时间、日期、主机名
set -g status-right "#[fg=cyan]%Y-%m-%d %H:%M:%S #[fg=green]#H"

# 窗口状态样式
setw -g window-status-style fg=cyan,bg=black
setw -g window-status-current-style fg=white,bg=blue,bold
setw -g window-status-format " #I:#W "
setw -g window-status-current-format " #I:#W "

# 面板边框样式
set -g pane-border-style fg=white
set -g pane-active-border-style fg=green

# 消息样式
set -g message-style fg=white,bg=black,bold

# =============================================
# 其他设置
# =============================================

# 设置终端标题
set -g set-titles on
set -g set-titles-string "#T"

# 自动重命名窗口
setw -g automatic-rename on
setw -g automatic-rename-format "#{pane_current_command}"

# 监控窗口活动
setw -g monitor-activity on
set -g visual-activity on

# 监控窗口静音
setw -g monitor-silence 0

# 设置焦点事件
set -g focus-events on

# 设置转义时间（解决 vim 延迟问题）
set -sg escape-time 0

# 设置重复超时时间
set -g repeat-time 1000
```

### 9.3 重新加载配置
```bash
# 在 Tmux 内部
Ctrl+b :source-file ~/.tmux.conf

# 或使用绑定的快捷键
Ctrl+b r

# 命令行
tmux source-file ~/.tmux.conf
```

---

## 十、高级用法

### 10.1 会话共享
```bash
# 用户 A 创建共享会话
tmux -S /tmp/shared new -s shared_session

# 设置权限让其他用户可以访问
chmod 777 /tmp/shared

# 用户 B 连接到共享会话
tmux -S /tmp/shared attach -t shared_session

# 或使用 sudo 切换用户后连接
sudo -u otheruser tmux -S /tmp/shared attach -t shared_session
```

### 10.2 保存和恢复会话
使用 `tmux-resurrect` 插件：

```bash
# 安装 tpm（Tmux 插件管理器）
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# 在 ~/.tmux.conf 中添加：
# List of plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# 自动保存和恢复
set -g @continuum-restore 'on'
set -g @continuum-save-interval '10'

# 初始化 TPM（必须放在配置文件最后）
run '~/.tmux/plugins/tpm/tpm'

# 安装插件
# 在 Tmux 中按 Ctrl+b I（大写 I）

# 手动保存会话
Ctrl+b Ctrl+s

# 手动恢复会话
Ctrl+b Ctrl+r
```

### 10.3 多服务器管理
```bash
# 创建脚本同时管理多台服务器
#!/bin/bash
# tmux-cluster.sh

SESSION="cluster"
tmux new-session -d -s $SESSION

# 分割窗口
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# 在每个面板中 SSH 到不同服务器
tmux send-keys -t 0 "ssh server1" C-m
tmux send-keys -t 1 "ssh server2" C-m
tmux send-keys -t 2 "ssh server3" C-m
tmux send-keys -t 3 "ssh server4" C-m

# 同步所有面板输入（广播模式）
tmux setw synchronize-panes on

tmux attach -t $SESSION
```

### 10.4 同步面板（广播模式）
```bash
# 在 Tmux 内部开启同步
Ctrl+b :setw synchronize-panes on

# 关闭同步
Ctrl+b :setw synchronize-panes off

# 或使用快捷键（需要配置）
# bind e setw synchronize-panes
```

### 10.5 日志记录
```bash
# 开始记录面板输出
Ctrl+b :pipe-pane -o "cat >> ~/#W.log" \; display "开始记录日志"

# 停止记录
Ctrl+b :pipe-pane \; display "停止记录日志"
```

---

## 十一、Tmux 插件

### 11.1 插件管理器（TPM）
```bash
# 安装 TPM
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# 在 ~/.tmux.conf 中添加：
set -g @plugin 'tmux-plugins/tpm'

# 初始化（放在配置文件最后）
run '~/.tmux/plugins/tpm/tpm'

# 安装插件：Ctrl+b I
# 更新插件：Ctrl+b U
# 卸载插件：在配置中删除后按 Ctrl+b Alt+u
```

### 11.2 推荐插件
```bash
# 会话保存和恢复
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# 复制到系统剪贴板
set -g @plugin 'tmux-plugins/tmux-yank'

# 快速打开文件/URL
set -g @plugin 'tmux-plugins/tmux-open'

# 电池状态显示
set -g @plugin 'tmux-plugins/tmux-battery'

# CPU 状态显示
set -g @plugin 'tmux-plugins/tmux-cpu'

# 在线状态显示
set -g @plugin 'tmux-plugins/tmux-online-status'

# 侧边栏文件浏览器
set -g @plugin 'tmux-plugins/tmux-sidebar'

# 主题
set -g @plugin 'jimeh/tmux-themepack'
set -g @themepack 'powerline/default/cyan'

# 会话管理器
set -g @plugin 'tmux-plugins/tmux-sessionist'
```

---

## 十二、实用技巧

### 12.1 快速启动常用布局
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加别名
alias tmux-dev='tmux new-session -d -s dev -n editor -c ~/projects \; \
    send-keys "vim" C-m \; \
    split-window -v -p 30 -c ~/projects \; \
    send-keys "npm run dev" C-m \; \
    new-window -n terminal -c ~/projects \; \
    select-window -t 1 \; \
    attach'

# 使用
# tmux-dev
```

### 12.2 与系统剪贴板集成
```bash
# Linux (xclip)
bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "xclip -selection clipboard"
bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "xclip -selection clipboard"

# macOS (pbcopy)
bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "pbcopy"

# WSL (clip.exe)
bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "clip.exe"
```

### 12.3 自动启动 Tmux
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
if command -v tmux > /dev/null 2>&1; then
    # 如果不在 Tmux 中，自动附加或创建会话
    if [ -z "$TMUX" ]; then
        tmux attach -t default || tmux new -s default
    fi
fi
```

### 12.4 Tmux 和 Vim 导航整合
```bash
# 使用 vim-tmux-navigator 插件
# 在 ~/.tmux.conf 中添加：

# Smart pane switching with awareness of Vim splits.
# See: https://github.com/christoomey/vim-tmux-navigator
is_vim="ps -o state= -o comm= -t '#{pane_tty}' \
    | grep -iqE '^[^TXZ ]+ +(\S+\/)?g?(view|n?vim?x?)(diff)?$'"
bind-key -n 'C-h' if-shell "$is_vim" 'send-keys C-h'  'select-pane -L'
bind-key -n 'C-j' if-shell "$is_vim" 'send-keys C-j'  'select-pane -D'
bind-key -n 'C-k' if-shell "$is_vim" 'send-keys C-k'  'select-pane -U'
bind-key -n 'C-l' if-shell "$is_vim" 'send-keys C-l'  'select-pane -R'
bind-key -n 'C-\' if-shell "$is_vim" 'send-keys C-\\' 'select-pane -l'
```

---

## 十三、常见问题

### 13.1 鼠标滚动不工作
```bash
# 确保启用了鼠标支持
set -g mouse on

# 在 Tmux 2.1+ 中，鼠标滚动会自动进入复制模式
# 如果不行，尝试按 Ctrl+b [ 进入复制模式后滚动
```

### 13.2 颜色显示不正确
```bash
# 确保设置了正确的终端类型
set -g default-terminal "screen-256color"
# 或
set -g default-terminal "tmux-256color"

# 在 ~/.bashrc 中也要设置
export TERM="xterm-256color"
```

### 13.3 Vim 在 Tmux 中响应慢
```bash
# 减少转义时间
set -sg escape-time 0
# 或
set -sg escape-time 10
```

### 13.4 复制模式无法使用
```bash
# 确保设置了正确的模式键
setw -g mode-keys vi

# 如果使用 Emacs 风格
setw -g mode-keys emacs
```

### 13.5 会话意外断开
```bash
# 检查 Tmux 服务器是否还在运行
tmux ls

# 如果没有输出，说明服务器已停止
# 检查日志（如果有配置的话）

# 使用 tmux-resurrect 插件可以自动保存和恢复会话
```

---

## 十四、Tmux 命令行工具

### 14.1 常用命令
```bash
# 启动新会话
tmux new -s session_name

# 列出会话
tmux ls

# 附加到会话
tmux attach -t session_name

# 关闭会话
tmux kill-session -t session_name

# 发送命令到会话
tmux send-keys -t session_name:0 "command" C-m

# 捕获面板内容
tmux capture-pane -t session_name:0.0 -p > output.txt

# 显示窗口信息
tmux display-message -p "Session: #S, Window: #I, Pane: #P"
```

### 14.2 格式字符串
Tmux 支持多种格式字符串用于自定义显示：

| 变量 | 说明 |
|------|------|
| `#S` | 会话名称 |
| `#W` | 窗口名称 |
| `#I` | 窗口索引 |
| `#P` | 面板索引 |
| `#T` | 面板标题 |
| `#F` | 窗口标志 |
| `#H` | 主机名 |
| `#h` | 主机名（短） |
| `#D` | 面板唯一 ID |
| `# pane_current_path` | 当前面板路径 |
| `# pane_current_command` | 当前面板命令 |

---

## 十五、总结

Tmux 是一个功能强大的终端复用工具，掌握它可以显著提升终端工作效率。关键要点：

1. **会话管理**：使用会话保持任务在后台运行
2. **窗口管理**：使用窗口组织不同的工作环境
3. **面板管理**：使用面板同时查看多个终端输出
4. **配置文件**：定制适合自己的快捷键和外观
5. **插件系统**：使用插件扩展 Tmux 功能

> **学习建议**：
> - 从基础开始，先掌握会话、窗口、面板的基本操作
> - 逐步配置 ~/.tmux.conf，定制适合自己的环境
> - 学习使用复制模式，方便在终端中复制文本
> - 探索插件系统，找到适合自己的工具组合
