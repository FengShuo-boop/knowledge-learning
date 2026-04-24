# Linux 常用命令详解

> 本文档涵盖 Linux 系统中最常用的命令，包含详细说明、常用选项和实际示例。适用于 CentOS/Ubuntu/Debian/Arch 等主流发行版。

---

## 目录

1. [文件和目录操作](#1-文件和目录操作)
2. [文本处理](#2-文本处理)
3. [系统管理](#3-系统管理)
4. [网络管理](#4-网络管理)
5. [压缩归档](#5-压缩归档)
6. [磁盘管理](#6-磁盘管理)
7. [软件包管理](#7-软件包管理)
8. [其他实用命令](#8-其他实用命令)
9. [命令速查表](#9-命令速查表)

---

## 1. 文件和目录操作

### 1.1 ls -- 列出目录内容

**说明**：显示目录中的文件和子目录列表。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-l` | 长格式显示（权限、所有者、大小、时间等） |
| `-a` | 显示所有文件，包括隐藏文件（以`.`开头的文件） |
| `-h` | 人类可读格式显示文件大小（K、M、G） |
| `-d` | 显示目录本身，而非目录内容 |
| `-t` | 按修改时间排序（最新的在前） |
| `-r` | 反向排序 |
| `-S` | 按文件大小排序（最大的在前） |
| `-R` | 递归显示子目录内容 |
| `-i` | 显示文件的 inode 号 |
| `--color` | 彩色输出 |

**示例**：

```bash
# 基本列出
ls

# 长格式详细信息
ls -l

# 显示隐藏文件
ls -la

# 人类可读的文件大小
ls -lh

# 按时间排序
ls -lt

# 递归列出所有文件
ls -R /var/log

# 只显示目录
ls -d */

# 组合使用：显示所有文件详细信息，按大小排序
ls -lahS
```

---

### 1.2 cd -- 切换目录

**说明**：更改当前工作目录。

**常用用法**：

| 用法 | 说明 |
|------|------|
| `cd 目录名` | 进入指定目录 |
| `cd ..` | 返回上级目录 |
| `cd ~` 或 `cd` | 返回用户主目录 |
| `cd -` | 返回上一次所在的目录 |
| `cd /` | 进入根目录 |

**示例**：

```bash
# 进入 /var/log 目录
cd /var/log

# 返回上级目录
cd ..

# 返回主目录
cd ~

# 快速切换
cd /etc && cd -   # 先进入 /etc，再返回原目录

# 进入用户主目录下的 Documents
cd ~/Documents
```

---

### 1.3 pwd -- 显示当前目录

**说明**：打印当前工作目录的完整路径。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-L` | 显示逻辑路径（包括符号链接） |
| `-P` | 显示物理路径（解析所有符号链接） |

**示例**：

```bash
# 显示当前路径
pwd
# 输出：/home/aero/projects

# 显示物理路径
pwd -P
```

---

### 1.4 mkdir -- 创建目录

**说明**：创建新的目录。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-p` | 递归创建目录（如果父目录不存在则自动创建） |
| `-v` | 显示创建过程 |
| `-m` | 设置目录权限 |

**示例**：

```bash
# 创建单个目录
mkdir mydir

# 递归创建多级目录
mkdir -p projects/web/app

# 创建并设置权限
mkdir -m 755 secure_dir

# 同时创建多个目录
mkdir dir1 dir2 dir3

# 显示创建过程
mkdir -pv a/b/c/d
```

---

### 1.5 rm -- 删除文件或目录

**说明**：删除文件或目录（**谨慎使用，删除后通常无法恢复**）。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-r` | 递归删除目录及其内容 |
| `-f` | 强制删除，不提示确认 |
| `-i` | 删除前询问确认 |
| `-v` | 显示删除过程 |
| `-d` | 删除空目录 |

**示例**：

```bash
# 删除单个文件
rm file.txt

# 删除多个文件
rm file1.txt file2.txt

# 递归删除目录
rm -r mydir/

# 强制递归删除（极度危险！）
rm -rf mydir/

# 安全删除（每次询问）
rm -i important.txt

# 删除空目录
rm -d empty_dir/

# 显示删除过程
rm -rv logs/
```

> **警告**：`rm -rf /` 会删除整个系统！使用时务必确认路径。

---

### 1.6 cp -- 复制文件或目录

**说明**：复制文件或目录。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-r` | 递归复制目录 |
| `-i` | 覆盖前询问 |
| `-f` | 强制覆盖 |
| `-v` | 显示复制过程 |
| `-p` | 保留文件属性（权限、时间戳等） |
| `-u` | 仅当源文件较新时才复制 |
| `-a` | 归档模式（相当于 `-dR --preserve=all`） |

**示例**：

```bash
# 复制文件
cp file.txt backup.txt

# 复制到目录
cp file.txt /backup/

# 递归复制目录
cp -r mydir/ backup/

# 保留属性复制
cp -p file.txt file.txt.bak

# 归档模式复制（保留所有属性，包括链接）
cp -a source_dir/ dest_dir/

# 显示复制过程
cp -rv src/ dst/

# 复制多个文件
cp file1.txt file2.txt /backup/
```

---

### 1.7 mv -- 移动或重命名文件/目录

**说明**：移动文件或目录，也可用于重命名。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-i` | 覆盖前询问 |
| `-f` | 强制覆盖 |
| `-v` | 显示移动过程 |
| `-u` | 仅当源文件较新时才移动 |
| `-n` | 不覆盖已存在的文件 |

**示例**：

```bash
# 重命名文件
mv oldname.txt newname.txt

# 移动文件到目录
mv file.txt /backup/

# 移动目录
mv mydir/ /backup/

# 移动多个文件
mv file1.txt file2.txt /backup/

# 覆盖前询问
mv -i file.txt /existing/

# 显示移动过程
mv -v *.txt /backup/
```

---

### 1.8 touch -- 创建空文件或修改时间戳

**说明**：创建空文件，或更新已有文件的访问和修改时间。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-a` | 仅修改访问时间 |
| `-m` | 仅修改修改时间 |
| `-t` | 指定时间戳（格式：[[CC]YY]MMDDhhmm[.ss]） |
| `-c` | 不创建不存在的文件 |
| `-r` | 使用参考文件的时间戳 |

**示例**：

```bash
# 创建空文件
touch newfile.txt

# 创建多个空文件
touch file1.txt file2.txt file3.txt

# 更新文件时间戳为当前时间
touch existing.txt

# 指定时间戳
touch -t 202401011200 file.txt

# 复制另一个文件的时间戳
touch -r reference.txt target.txt

# 批量创建文件
touch file{1..10}.txt
```

---

### 1.9 cat -- 连接并显示文件内容

**说明**：显示文件内容，或连接多个文件。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-n` | 显示行号 |
| `-b` | 显示行号（空行不编号） |
| `-s` | 压缩多个空行为一个 |
| `-E` | 在行尾显示 `$` |
| `-T` | 将制表符显示为 `^I` |

**示例**：

```bash
# 显示文件内容
cat file.txt

# 显示行号
cat -n file.txt

# 连接多个文件
cat file1.txt file2.txt > combined.txt

# 追加内容到文件
cat >> file.txt << EOF
这是追加的内容
EOF

# 创建文件
cat > newfile.txt << EOF
第一行
第二行
EOF

# 显示制表符和行尾
cat -A file.txt
```

---

### 1.10 less -- 分页查看文件

**说明**：分页查看文件内容，支持前后翻页（比 `more` 更强大）。

**常用操作**：

| 按键 | 说明 |
|------|------|
| `Space` / `f` | 向下翻页 |
| `b` | 向上翻页 |
| `↓` / `j` | 向下移动一行 |
| `↑` / `k` | 向上移动一行 |
| `g` | 跳到文件开头 |
| `G` | 跳到文件末尾 |
| `/pattern` | 向下搜索 |
| `?pattern` | 向上搜索 |
| `n` | 下一个匹配 |
| `N` | 上一个匹配 |
| `q` | 退出 |

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-N` | 显示行号 |
| `-i` | 忽略大小写搜索 |
| `-S` | 不换行显示长行 |
| `+F` | 实时监控文件（类似 `tail -f`） |

**示例**：

```bash
# 查看文件
less /var/log/syslog

# 显示行号
less -N file.txt

# 忽略大小写搜索
less -i file.txt

# 查看命令输出
ps aux | less

# 实时监控文件
less +F /var/log/nginx/access.log
```

---

### 1.11 head -- 显示文件开头

**说明**：显示文件开头部分（默认前 10 行）。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-n N` | 显示前 N 行 |
| `-c N` | 显示前 N 字节 |
| `-q` | 不显示文件名（多文件时） |
| `-v` | 总是显示文件名 |

**示例**：

```bash
# 显示前 10 行
head file.txt

# 显示前 20 行
head -n 20 file.txt

# 显示前 100 字节
head -c 100 file.txt

# 查看多个文件
head file1.txt file2.txt

# 查看命令输出前 5 行
ps aux | head -n 5
```

---

### 1.12 tail -- 显示文件末尾

**说明**：显示文件末尾部分（默认后 10 行），常用于查看日志。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-n N` | 显示后 N 行 |
| `-c N` | 显示后 N 字节 |
| `-f` | 实时监控文件变化（follow） |
| `-F` | 实时监控，并在文件轮转时重新打开 |
| `--pid=PID` | 当指定进程结束时退出 |

**示例**：

```bash
# 显示后 10 行
tail /var/log/syslog

# 显示后 50 行
tail -n 50 file.txt

# 实时监控日志（最常用）
tail -f /var/log/nginx/access.log

# 实时监控并显示行号
tail -f -n +1 /var/log/app.log

# 从第 100 行开始显示
tail -n +100 file.txt

# 监控多个文件
tail -f /var/log/*.log
```

---

### 1.13 find -- 查找文件

**说明**：在目录层次结构中搜索文件。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-name` | 按文件名查找（区分大小写） |
| `-iname` | 按文件名查找（不区分大小写） |
| `-type` | 按文件类型查找（f=文件, d=目录, l=链接） |
| `-size` | 按文件大小查找 |
| `-mtime` | 按修改时间查找（天） |
| `-mmin` | 按修改时间查找（分钟） |
| `-perm` | 按权限查找 |
| `-user` | 按所有者查找 |
| `-group` | 按所属组查找 |
| `-exec` | 对找到的文件执行命令 |
| `-delete` | 删除找到的文件 |
| `-maxdepth` | 最大搜索深度 |
| `-mindepth` | 最小搜索深度 |

**示例**：

```bash
# 按名称查找
find /home -name "*.txt"

# 不区分大小写查找
find / -iname "*config*"

# 查找目录
find /var -type d -name "log"

# 查找大于 100MB 的文件
find / -size +100M

# 查找 7 天内修改过的文件
find . -mtime -7

# 查找 30 天前的文件并删除
find /tmp -mtime +30 -delete

# 查找并执行命令
find . -name "*.log" -exec ls -lh {} \;

# 查找并复制
find . -name "*.conf" -exec cp {} /backup/ \;

# 限制搜索深度
find /etc -maxdepth 2 -name "*.conf"

# 查找空文件/目录
find . -empty

# 查找并修改权限
find . -type f -perm 644 -exec chmod 664 {} \;

# 按用户查找
find /home -user aero

# 组合条件查找
find . -name "*.log" -mtime +7 -size +10M
```

---

### 1.14 chmod -- 修改文件权限

**说明**：更改文件或目录的访问权限。

**权限表示**：

| 权限 | 数字 | 说明 |
|------|------|------|
| r（读） | 4 | 读取文件内容 / 列出目录内容 |
| w（写） | 2 | 修改文件 / 在目录中创建删除文件 |
| x（执行） | 1 | 执行文件 / 进入目录 |

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-R` | 递归修改 |
| `-v` | 显示修改过程 |
| `--reference` | 参考其他文件权限 |

**示例**：

```bash
# 数字模式设置权限
chmod 755 script.sh      # rwxr-xr-x
chmod 644 file.txt       # rw-r--r--
chmod 700 private.key    # rwx------

# 符号模式
chmod u+x script.sh      # 给所有者添加执行权限
chmod g-w file.txt       # 去掉组的写权限
chmod o+r file.txt       # 给其他人添加读权限
chmod a+x script.sh      # 给所有人添加执行权限
chmod u=rwx,g=rx,o=r file.txt

# 递归修改目录权限
chmod -R 755 mydir/

# 参考其他文件权限
chmod --reference=file1.txt file2.txt

# 显示修改过程
chmod -v 755 script.sh
```

---

### 1.15 chown -- 修改文件所有者

**说明**：更改文件或目录的所有者和所属组。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-R` | 递归修改 |
| `-v` | 显示修改过程 |
| `--reference` | 参考其他文件 |

**示例**：

```bash
# 修改所有者
chown aero file.txt

# 同时修改所有者和组
chown aero:users file.txt

# 只修改组
chown :developers file.txt

# 递归修改目录
chown -R aero:aero mydir/

# 参考其他文件
chown --reference=file1.txt file2.txt

# 显示修改过程
chown -v aero file.txt
```

---

### 1.16 ln -- 创建链接

**说明**：创建硬链接或符号链接（软链接）。

| 类型 | 说明 | 特点 |
|------|------|------|
| 硬链接 | 指向文件的 inode | 不能跨文件系统，不能链接目录，删除源文件不影响 |
| 软链接 | 指向文件路径 | 可以跨文件系统，可以链接目录，源文件删除后失效 |

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-s` | 创建符号链接（软链接） |
| `-f` | 强制创建（覆盖已有链接） |
| `-i` | 覆盖前询问 |
| `-v` | 显示创建过程 |

**示例**：

```bash
# 创建硬链接
ln file.txt hardlink.txt

# 创建软链接（最常用）
ln -s /var/log/nginx nginx-logs

# 创建目录软链接
ln -s /opt/myapp/bin myapp

# 强制创建/更新链接
ln -sf /new/path/to/file linkname

# 创建链接到目录
ln -s /home/aero/Documents docs

# 显示创建过程
ln -sv /var/log loglink
```


---

## 2. 文本处理

### 2.1 grep -- 文本搜索

**说明**：在文件中搜索匹配正则表达式的行。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-i` | 忽略大小写 |
| `-v` | 反向匹配（显示不匹配的行） |
| `-n` | 显示行号 |
| `-r` / `-R` | 递归搜索目录 |
| `-l` | 只显示包含匹配的文件名 |
| `-c` | 显示匹配行数 |
| `-w` | 匹配整个单词 |
| `-E` | 使用扩展正则表达式（等同于 `egrep`） |
| `-F` | 固定字符串匹配（等同于 `fgrep`） |
| `-o` | 只显示匹配的部分 |
| `-A N` | 显示匹配行及后 N 行 |
| `-B N` | 显示匹配行及前 N 行 |
| `-C N` | 显示匹配行及前后 N 行 |
| `--color` | 高亮显示匹配内容 |
| `-e` | 指定多个模式 |

**示例**：

```bash
# 基本搜索
grep "error" log.txt

# 忽略大小写
grep -i "error" log.txt

# 显示行号
grep -n "error" log.txt

# 递归搜索
grep -r "TODO" /home/aero/projects/

# 只显示文件名
grep -rl "error" /var/log/

# 反向匹配
grep -v "DEBUG" log.txt

# 匹配整个单词
grep -w "error" log.txt

# 显示上下文
grep -C 3 "Exception" app.log

# 多个模式
grep -e "error" -e "ERROR" -e "Error" log.txt

# 使用正则表达式
grep -E "error|ERROR|Error" log.txt

# 只显示匹配部分
grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+" access.log

# 搜索并统计
grep -c "error" log.txt

# 管道中使用
ps aux | grep nginx

# 排除自身
grep "nginx" | grep -v grep
# 或
grep "[n]ginx"

# 高亮显示
grep --color=auto "error" log.txt
```

---

### 2.2 sed -- 流编辑器

**说明**：对文本进行过滤和转换，常用于批量替换。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-i` | 直接修改文件 |
| `-e` | 执行多个命令 |
| `-n` | 静默模式（只打印指定行） |
| `-r` | 使用扩展正则表达式 |

**常用命令**：

| 命令 | 说明 |
|------|------|
| `s/old/new/` | 替换 |
| `d` | 删除行 |
| `p` | 打印行 |
| `i` | 行前插入 |
| `a` | 行后追加 |
| `c` | 替换整行 |
| `y/old/new/` | 字符转换 |

**示例**：

```bash
# 基本替换（输出到屏幕，不修改原文件）
sed 's/foo/bar/' file.txt

# 全局替换
sed 's/foo/bar/g' file.txt

# 直接修改文件
sed -i 's/foo/bar/g' file.txt

# 修改并备份
sed -i.bak 's/foo/bar/g' file.txt

# 替换第 2 次出现的匹配
sed 's/foo/bar/2' file.txt

# 替换第 2 次及之后的所有匹配
sed 's/foo/bar/2g' file.txt

# 删除空行
sed '/^$/d' file.txt

# 删除第 5 行
sed '5d' file.txt

# 删除 1-3 行
sed '1,3d' file.txt

# 打印第 5 行
sed -n '5p' file.txt

# 打印 5-10 行
sed -n '5,10p' file.txt

# 在第 5 行前插入
sed '5i\This is new line' file.txt

# 在第 5 行后追加
sed '5a\This is appended' file.txt

# 多个命令
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt

# 使用正则表达式
sed -r 's/([0-9]+)/NUMBER:\1/g' file.txt

# 删除行尾空格
sed 's/[[:space:]]*$//' file.txt

# 在文件开头添加一行
sed '1i\# Header' file.txt

# 在文件末尾添加一行
sed '$a\# Footer' file.txt

# 只替换包含特定模式的行
sed '/pattern/s/foo/bar/g' file.txt
```

---

### 2.3 awk -- 文本处理工具

**说明**：强大的文本处理语言，适合处理结构化数据（如 CSV、日志等）。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-F` | 指定字段分隔符 |
| `-v` | 定义变量 |
| `-f` | 从文件读取 awk 脚本 |

**内置变量**：

| 变量 | 说明 |
|------|------|
| `$0` | 整行 |
| `$1, $2, ...` | 第 1、2... 个字段 |
| `NF` | 字段数量 |
| `NR` | 行号 |
| `FS` | 字段分隔符 |
| `OFS` | 输出字段分隔符 |
| `RS` | 记录分隔符 |
| `ORS` | 输出记录分隔符 |

**示例**：

```bash
# 打印第一列
awk '{print $1}' file.txt

# 指定分隔符（如冒号）
awk -F: '{print $1}' /etc/passwd

# 打印多列
awk '{print $1, $3}' file.txt

# 打印最后一列
awk '{print $NF}' file.txt

# 带行号打印
awk '{print NR, $0}' file.txt

# 条件过滤
awk '$3 > 100 {print $0}' file.txt

# 多条件
awk '$2 == "error" && $3 > 10 {print $0}' log.txt

# 计算总和
awk '{sum += $1} END {print sum}' numbers.txt

# 计算平均值
awk '{sum += $1; count++} END {print sum/count}' numbers.txt

# 查找最大最小值
awk 'NR==1{min=max=$1} {if($1<min)min=$1; if($1>max)max=$1} END{print "min:", min, "max:", max}' numbers.txt

# 格式化输出
awk '{printf "Name: %-10s Age: %3d\n", $1, $2}' data.txt

# 替换分隔符输出
awk -F: -v OFS=',' '{print $1, $3, $6}' /etc/passwd

# 去重（保留第一个）
awk '!seen[$0]++' file.txt

# 统计词频
awk '{for(i=1;i<=NF;i++) count[$i]++} END {for(word in count) print count[word], word}' words.txt

# 处理 CSV
awk -F',' '{print "Name:", $1, "Email:", $2}' users.csv
```

---

### 2.4 sort -- 文本排序

**说明**：对文本行进行排序。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-r` | 反向排序 |
| `-n` | 按数字排序 |
| `-k N` | 按第 N 列排序 |
| `-t` | 指定分隔符 |
| `-u` | 去重（唯一） |
| `-f` | 忽略大小写 |
| `-M` | 按月份排序 |
| `-h` | 按人类可读大小排序 |
| `-R` | 随机排序 |

**示例**：

```bash
# 基本排序
sort file.txt

# 反向排序
sort -r file.txt

# 按数字排序
sort -n numbers.txt

# 按第 2 列排序
sort -k2 file.txt

# 按第 3 列数字排序
sort -k3 -n file.txt

# 指定分隔符后排序
sort -t: -k3 -n /etc/passwd

# 去重排序
sort -u file.txt

# 忽略大小写
sort -f file.txt

# 按文件大小排序
ls -lh | sort -k5 -h

# 随机排序
sort -R file.txt

# 复杂排序：先按第 2 列，再按第 1 列
sort -k2,2 -k1,1 file.txt
```

---

### 2.5 uniq -- 去重

**说明**：过滤或报告重复的行（**需要先排序**）。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-c` | 统计重复次数 |
| `-d` | 只显示重复的行 |
| `-u` | 只显示唯一的行 |
| `-i` | 忽略大小写 |
| `-f N` | 跳过前 N 个字段 |
| `-s N` | 跳过前 N 个字符 |
| `-w N` | 只比较前 N 个字符 |

**示例**：

```bash
# 去重（需要先 sort）
sort file.txt | uniq

# 统计每行出现次数
sort file.txt | uniq -c

# 只显示重复的行
sort file.txt | uniq -d

# 只显示唯一的行
sort file.txt | uniq -u

# 忽略大小写
sort file.txt | uniq -i

# 统计访问最多的 IP
cut -d' ' -f1 access.log | sort | uniq -c | sort -rn | head

# 跳过前 2 个字段比较
sort file.txt | uniq -f 2
```

---

### 2.6 cut -- 提取字段

**说明**：从每行中提取指定部分。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-f` | 指定字段 |
| `-d` | 指定分隔符 |
| `-c` | 按字符位置提取 |
| `--complement` | 提取指定部分之外的内容 |

**示例**：

```bash
# 提取第 1 列（默认分隔符为 Tab）
cut -f1 file.txt

# 指定分隔符提取列
cut -d: -f1 /etc/passwd

# 提取多个列
cut -d: -f1,3,6 /etc/passwd

# 提取列范围
cut -d: -f1-3 /etc/passwd

# 按字符位置提取
cut -c1-10 file.txt

# 提取第 5 个字符之后
cut -c5- file.txt

# 排除第 2 列
cut -f2 --complement file.txt

# 提取 IP 地址
cut -d' ' -f1 access.log
```

---

### 2.7 tr -- 字符转换

**说明**：转换或删除字符。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-d` | 删除字符 |
| `-s` | 压缩连续重复的字符 |
| `-c` | 取反（补集） |

**示例**：

```bash
# 小写转大写
echo "hello" | tr 'a-z' 'A-Z'

# 大写转小写
echo "HELLO" | tr 'A-Z' 'a-z'

# 删除字符
echo "hello 123 world" | tr -d '0-9'

# 压缩空格
echo "hello    world" | tr -s ' '

# 替换多个字符
echo "hello" | tr 'aeiou' '12345'

# 删除换行（合并多行）
cat file.txt | tr -d '\n'

# 替换特定字符
echo "hello:world" | tr ':' ' '

# 删除非数字字符
echo "abc123def456" | tr -cd '0-9'

# 生成随机密码
tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 16
```

---

### 2.8 wc -- 统计行数/单词数/字节数

**说明**：统计文件中的行数、单词数和字节数。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-l` | 统计行数 |
| `-w` | 统计单词数 |
| `-c` | 统计字节数 |
| `-m` | 统计字符数 |
| `-L` | 显示最长行的长度 |

**示例**：

```bash
# 统计所有信息
wc file.txt

# 只统计行数
wc -l file.txt

# 统计单词数
wc -w file.txt

# 统计字节数
wc -c file.txt

# 统计多个文件
wc *.txt

# 统计目录下文件数量
ls | wc -l

# 统计代码行数
find . -name "*.py" | xargs wc -l

# 统计进程数量
ps aux | wc -l

# 显示最长行
wc -L file.txt
```

---

### 2.9 diff -- 比较文件差异

**说明**：比较两个文件或目录的差异。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-u` | 统一格式输出（推荐） |
| `-c` | 上下文格式输出 |
| `-r` | 递归比较目录 |
| `-i` | 忽略大小写 |
| `-w` | 忽略空白字符 |
| `-B` | 忽略空行 |
| `-q` | 只报告是否不同 |
| `-y` | 并排显示 |

**示例**：

```bash
# 基本比较
diff file1.txt file2.txt

# 统一格式（最常用）
diff -u file1.txt file2.txt

# 生成补丁文件
diff -u file1.txt file2.txt > patch.diff

# 应用补丁
patch file1.txt < patch.diff

# 递归比较目录
diff -r dir1/ dir2/

# 忽略大小写
diff -i file1.txt file2.txt

# 忽略空白
diff -w file1.txt file2.txt

# 并排显示
diff -y file1.txt file2.txt

# 只报告是否不同
diff -q file1.txt file2.txt

# 比较并显示颜色
diff --color=auto file1.txt file2.txt
```


---

## 3. 系统管理

### 3.1 ps -- 查看进程状态

**说明**：显示当前进程的快照。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `aux` | 显示所有用户的所有进程（BSD 风格） |
| `-ef` | 显示所有进程的完整信息（System V 风格） |
| `-u 用户` | 显示指定用户的进程 |
| `-p PID` | 显示指定 PID 的进程 |
| `--sort` | 按指定字段排序 |
| `-o` | 自定义输出格式 |
| `-H` | 显示进程树 |
| `-L` | 显示线程 |

**示例**：

```bash
# 显示所有进程（BSD 风格）
ps aux

# 显示所有进程（System V 风格）
ps -ef

# 查找特定进程
ps aux | grep nginx

# 显示指定用户的进程
ps -u aero

# 显示指定 PID 的进程
ps -p 1234

# 显示进程树
ps auxf
ps -efH

# 按 CPU 使用率排序
ps aux --sort=-%cpu | head

# 按内存使用率排序
ps aux --sort=-%mem | head

# 自定义输出
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head

# 显示线程
ps -eLf | grep nginx
```

---

### 3.2 top -- 实时进程监控

**说明**：实时显示系统中各个进程的资源占用情况。

**常用交互命令**：

| 按键 | 说明 |
|------|------|
| `q` | 退出 |
| `Space` | 立即刷新 |
| `k` | 杀死进程（输入 PID） |
| `r` | 修改进程优先级（renice） |
| `M` | 按内存排序 |
| `P` | 按 CPU 排序 |
| `T` | 按运行时间排序 |
| `1` | 显示每个 CPU 核心 |
| `c` | 显示完整命令行 |
| `u` | 按用户过滤 |
| `H` | 显示线程 |
| `d` | 修改刷新间隔 |

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-d N` | 设置刷新间隔为 N 秒 |
| `-p PID` | 监控指定进程 |
| `-u 用户` | 监控指定用户 |
| `-n N` | 刷新 N 次后退出 |
| `-b` | 批处理模式（用于脚本） |

**示例**：

```bash
# 基本使用
top

# 设置刷新间隔为 2 秒
top -d 2

# 监控指定进程
top -p 1234

# 监控指定用户
top -u aero

# 批处理模式，输出 5 次
top -b -n 5 > top.log

# 按内存排序后批处理输出
top -b -n 1 -o %MEM | head -20
```

---

### 3.3 kill -- 终止进程

**说明**：向进程发送信号，默认发送终止信号（SIGTERM）。

**常用信号**：

| 信号 | 数值 | 说明 |
|------|------|------|
| `SIGHUP` | 1 | 挂起信号，常用于重新加载配置 |
| `SIGINT` | 2 | 中断信号（Ctrl+C） |
| `SIGKILL` | 9 | 强制终止，无法捕获或忽略 |
| `SIGTERM` | 15 | 正常终止（默认） |
| `SIGUSR1` | 10 | 用户自定义信号 1 |
| `SIGUSR2` | 12 | 用户自定义信号 2 |

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-l` | 列出所有信号 |
| `-s 信号` | 指定信号 |
| `-9` | 强制终止 |

**示例**：

```bash
# 正常终止进程
kill 1234

# 强制终止进程
kill -9 1234

# 发送指定信号
kill -HUP 1234      # 重新加载配置
kill -TERM 1234     # 正常终止
kill -INT 1234      # 中断

# 终止多个进程
kill 1234 5678 9012

# 根据进程名终止（使用 killall）
killall nginx
killall -9 firefox

# 根据模式终止（使用 pkill）
pkill -f "python app.py"
pkill -u aero
```

---

### 3.4 df -- 查看磁盘空间

**说明**：显示文件系统的磁盘空间使用情况。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-h` | 人类可读格式 |
| `-T` | 显示文件系统类型 |
| `-i` | 显示 inode 使用情况 |
| `-a` | 显示所有文件系统 |
| `-x 类型` | 排除指定类型的文件系统 |

**示例**：

```bash
# 基本查看
df

# 人类可读格式
df -h

# 显示文件系统类型
df -Th

# 查看 inode 使用
df -i

# 排除 tmpfs
df -h -x tmpfs -x devtmpfs

# 查看指定目录所在文件系统
df -h /home
```

---

### 3.5 du -- 查看目录/文件大小

**说明**：估算文件和目录的磁盘使用空间。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-h` | 人类可读格式 |
| `-s` | 只显示总计 |
| `-a` | 显示所有文件和目录 |
| `-c` | 显示总计 |
| `--max-depth=N` | 最大显示深度 |
| `-k` / `-m` | 以 KB / MB 显示 |
| `--exclude=模式` | 排除匹配的文件 |

**示例**：

```bash
# 查看当前目录大小
du -sh

# 查看指定目录大小
du -sh /var/log

# 查看当前目录下各子目录大小
du -h --max-depth=1

# 查看所有文件和目录大小
du -ah

# 显示总计
du -ch *.log

# 排除某些文件
du -sh --exclude="*.log" /var/log

# 查找最大的 10 个目录
du -h --max-depth=1 | sort -rh | head -10

# 查看当前目录下文件大小
du -sh *
```

---

### 3.6 free -- 查看内存使用

**说明**：显示系统内存使用情况。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-h` | 人类可读格式 |
| `-m` | 以 MB 显示 |
| `-g` | 以 GB 显示 |
| `-s N` | 每 N 秒刷新一次 |
| `-t` | 显示总计 |

**示例**：

```bash
# 基本查看
free

# 人类可读格式
free -h

# 以 MB 显示
free -m

# 持续监控
free -hs 2

# 显示总计
free -ht
```

---

### 3.7 uname -- 查看系统信息

**说明**：显示系统内核信息。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-a` | 显示所有信息 |
| `-s` | 显示内核名称 |
| `-n` | 显示主机名 |
| `-r` | 显示内核版本 |
| `-v` | 显示内核发布日期 |
| `-m` | 显示机器硬件名 |
| `-p` | 显示处理器类型 |
| `-i` | 显示硬件平台 |
| `-o` | 显示操作系统 |

**示例**：

```bash
# 显示所有信息
uname -a

# 显示内核版本
uname -r

# 显示主机名
uname -n

# 显示操作系统
uname -o

# 显示架构
uname -m
```

---

### 3.8 uptime -- 查看系统运行时间

**说明**：显示系统运行时间、当前用户数、系统负载。

**示例**：

```bash
# 基本查看
uptime

# 输出示例：
# 14:30:00 up 5 days, 2:15, 3 users,  load average: 0.52, 0.58, 0.59
```

**负载说明**：
- 三个数字分别表示 1 分钟、5 分钟、15 分钟的平均负载
- 负载值接近 CPU 核心数表示系统满负荷运行
- 超过 CPU 核心数表示系统过载

---

### 3.9 who -- 查看登录用户

**说明**：显示当前登录系统的用户。

**相关命令**：

| 命令 | 说明 |
|------|------|
| `who` | 显示登录用户 |
| `whoami` | 显示当前用户名 |
| `w` | 显示登录用户及其正在执行的命令 |
| `last` | 显示最近登录记录 |
| `lastlog` | 显示所有用户的最后登录时间 |

**示例**：

```bash
# 显示登录用户
who

# 显示当前用户
whoami

# 显示详细信息
w

# 显示最近登录
last

# 显示指定用户的登录记录
last aero

# 显示所有用户的最后登录
lastlog

# 显示未登录的用户
lastlog | grep "Never"
```

---

### 3.10 crontab -- 定时任务管理

**说明**：管理用户的定时任务。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-e` | 编辑当前用户的 crontab |
| `-l` | 列出当前用户的 crontab |
| `-r` | 删除当前用户的 crontab |
| `-u 用户` | 管理指定用户的 crontab（root 权限） |

**时间格式**：

```
* * * * * 命令
│ │ │ │ │
│ │ │ │ └── 星期 (0-7, 0 和 7 都表示周日)
│ │ │ └──── 月份 (1-12)
│ │ └────── 日期 (1-31)
│ └──────── 小时 (0-23)
└────────── 分钟 (0-59)
```

**特殊字符串**：

| 字符串 | 说明 |
|--------|------|
| `@reboot` | 系统启动时执行 |
| `@yearly` / `@annually` | 每年 1 月 1 日 0:00 |
| `@monthly` | 每月 1 日 0:00 |
| `@weekly` | 每周日 0:00 |
| `@daily` / `@midnight` | 每天 0:00 |
| `@hourly` | 每小时开始时 |

**示例**：

```bash
# 编辑 crontab
crontab -e

# 列出 crontab
crontab -l

# 删除 crontab
crontab -r

# 查看 root 的 crontab
sudo crontab -u root -l
```

**crontab 示例**：

```bash
# 每分钟执行
* * * * * /path/to/script.sh

# 每 5 分钟执行
*/5 * * * * /path/to/script.sh

# 每小时执行
0 * * * * /path/to/script.sh

# 每天凌晨 2 点执行
0 2 * * * /path/to/backup.sh

# 每周一凌晨 3 点执行
0 3 * * 1 /path/to/weekly.sh

# 每月 1 日凌晨执行
0 0 1 * * /path/to/monthly.sh

# 工作日每 2 小时执行
0 */2 * * 1-5 /path/to/workday.sh

# 系统启动时执行
@reboot /path/to/startup.sh

# 将输出重定向到日志
0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1
```

---

### 3.11 systemctl -- 系统服务管理

**说明**：管理系统服务（systemd）。

**常用命令**：

| 命令 | 说明 |
|------|------|
| `start 服务` | 启动服务 |
| `stop 服务` | 停止服务 |
| `restart 服务` | 重启服务 |
| `reload 服务` | 重新加载配置 |
| `status 服务` | 查看服务状态 |
| `enable 服务` | 开机自启 |
| `disable 服务` | 取消开机自启 |
| `is-enabled 服务` | 检查是否开机自启 |
| `list-units` | 列出所有已加载的单元 |
| `list-unit-files` | 列出所有单元文件 |
| `daemon-reload` | 重新加载 systemd 配置 |

**示例**：

```bash
# 启动服务
sudo systemctl start nginx

# 停止服务
sudo systemctl stop nginx

# 重启服务
sudo systemctl restart nginx

# 重新加载配置
sudo systemctl reload nginx

# 查看状态
sudo systemctl status nginx

# 开机自启
sudo systemctl enable nginx

# 取消开机自启
sudo systemctl disable nginx

# 查看所有运行中的服务
systemctl list-units --type=service --state=running

# 查看所有失败的服务
systemctl --failed

# 重新加载 systemd
sudo systemctl daemon-reload

# 查看服务日志
sudo journalctl -u nginx

# 查看实时日志
sudo journalctl -u nginx -f
```


---

## 4. 网络管理

### 4.1 ping -- 测试网络连通性

**说明**：向目标主机发送 ICMP 回显请求，测试网络连通性。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-c N` | 发送 N 个数据包后停止 |
| `-i N` | 设置发送间隔为 N 秒 |
| `-s N` | 设置数据包大小为 N 字节 |
| `-t N` | 设置 TTL（生存时间） |
| `-W N` | 设置超时时间为 N 秒 |
| `-q` | 静默模式，只显示摘要 |
| `-4` / `-6` | 强制使用 IPv4 / IPv6 |

**示例**：

```bash
# 基本测试（Linux 默认持续 ping，按 Ctrl+C 停止）
ping www.baidu.com

# 发送 4 个数据包
ping -c 4 www.baidu.com

# 设置数据包大小
ping -s 1024 www.baidu.com

# 设置间隔为 2 秒
ping -i 2 www.baidu.com

# 静默模式
ping -c 4 -q www.baidu.com

# 测试本地网关
ping -c 4 192.168.1.1
```

---

### 4.2 curl -- 数据传输工具

**说明**：命令行下传输数据的工具，支持多种协议。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-o 文件` | 输出到文件 |
| `-O` | 保留远程文件名 |
| `-L` | 跟随重定向 |
| `-I` | 只获取响应头 |
| `-v` | 显示详细过程 |
| `-s` | 静默模式 |
| `-S` | 显示错误 |
| `-X 方法` | 指定请求方法 |
| `-H "头信息"` | 添加请求头 |
| `-d "数据"` | 发送 POST 数据 |
| `-F "name=@file"` | 上传文件 |
| `-u 用户:密码` | 设置认证信息 |
| `-k` | 忽略 SSL 证书验证 |
| `--connect-timeout N` | 连接超时 |
| `--max-time N` | 最大传输时间 |

**示例**：

```bash
# 获取网页内容
curl https://www.example.com

# 保存到文件
curl -o page.html https://www.example.com

# 保留远程文件名
curl -O https://www.example.com/file.zip

# 跟随重定向
curl -L https://bit.ly/xxx

# 只获取响应头
curl -I https://www.example.com

# GET 请求
curl -X GET https://api.example.com/users

# POST 请求（表单数据）
curl -X POST -d "name=john&age=30" https://api.example.com/users

# POST 请求（JSON 数据）
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"john","age":30}' \
  https://api.example.com/users

# 上传文件
curl -F "file=@/path/to/file.zip" https://api.example.com/upload

# 基本认证
curl -u username:password https://api.example.com/data

# Bearer Token 认证
curl -H "Authorization: Bearer token123" https://api.example.com/data

# 下载并显示进度
curl -# -o file.zip https://example.com/file.zip

# 断点续传
curl -C - -o file.zip https://example.com/file.zip

# 忽略证书验证
curl -k https://self-signed.example.com
```

---

### 4.3 wget -- 文件下载工具

**说明**：从网络下载文件。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-O 文件` | 指定输出文件名 |
| `-P 目录` | 指定下载目录 |
| `-c` | 断点续传 |
| `-b` | 后台下载 |
| `-q` | 静默模式 |
| `--limit-rate=速度` | 限制下载速度 |
| `-t N` | 重试 N 次 |
| `-T N` | 超时 N 秒 |
| `--user=用户` | 设置用户名 |
| `--password=密码` | 设置密码 |
| `--no-check-certificate` | 忽略证书验证 |
| `-r` | 递归下载 |
| `-l N` | 递归深度 N |
| `-np` | 不向上级目录递归 |
| `-k` | 转换链接为本地链接 |
| `-m` | 镜像网站 |

**示例**：

```bash
# 基本下载
wget https://www.example.com/file.zip

# 指定输出文件名
wget -O myfile.zip https://www.example.com/file.zip

# 指定下载目录
wget -P /downloads https://www.example.com/file.zip

# 断点续传
wget -c https://www.example.com/large-file.zip

# 后台下载
wget -b https://www.example.com/large-file.zip

# 限制下载速度（100KB/s）
wget --limit-rate=100k https://www.example.com/file.zip

# 重试 5 次
wget -t 5 https://www.example.com/file.zip

# 镜像网站
wget -m -k -np https://www.example.com/docs/

# 下载整个目录
wget -r -np -nH --cut-dirs=2 https://www.example.com/files/

# 使用认证
wget --user=admin --password=secret https://www.example.com/protected/file.zip
```

---

### 4.4 netstat -- 网络状态查看

**说明**：显示网络连接、路由表、接口统计等。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-a` | 显示所有连接和监听端口 |
| `-t` | 显示 TCP 连接 |
| `-u` | 显示 UDP 连接 |
| `-n` | 显示数字地址（不解析主机名） |
| `-l` | 只显示监听端口 |
| `-p` | 显示进程信息 |
| `-r` | 显示路由表 |
| `-s` | 显示统计信息 |
| `-i` | 显示网络接口统计 |
| `-c` | 持续输出 |

**示例**：

```bash
# 显示所有连接
netstat -a

# 显示所有 TCP 连接（数字格式）
netstat -tan

# 显示所有监听端口（带进程信息）
netstat -tlnp

# 显示所有 UDP 监听端口
netstat -ulnp

# 显示路由表
netstat -rn

# 显示接口统计
netstat -i

# 持续显示 TCP 连接
netstat -tan -c 2

# 查找指定端口的进程
netstat -tlnp | grep :80
```

---

### 4.5 ss -- 套接字统计（netstat 的替代）

**说明**：比 netstat 更快更强大的套接字查看工具。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-t` | 显示 TCP 套接字 |
| `-u` | 显示 UDP 套接字 |
| `-l` | 只显示监听套接字 |
| `-a` | 显示所有套接字 |
| `-n` | 不解析服务名 |
| `-p` | 显示进程信息 |
| `-s` | 显示统计信息 |
| `-4` / `-6` | 只显示 IPv4 / IPv6 |
| `-o` | 显示计时器信息 |
| `-i` | 显示内部 TCP 信息 |

**示例**：

```bash
# 显示所有 TCP 连接
ss -t -a

# 显示所有监听端口（带进程）
ss -tlnp

# 显示所有 UDP 监听端口
ss -ulnp

# 显示所有连接（简要）
ss -s

# 查找指定端口的连接
ss -tlnp | grep :80

# 显示所有连接到 22 端口的进程
ss -tnp | grep :22

# 按状态过滤
ss -t state established

# 显示内存使用情况
ss -m
```

---

### 4.6 ip -- 网络配置工具

**说明**：强大的网络配置和查看工具（ifconfig 的替代）。

**常用子命令**：

| 子命令 | 说明 |
|--------|------|
| `addr` / `a` | 管理 IP 地址 |
| `link` / `l` | 管理网络接口 |
| `route` / `r` | 管理路由 |
| `neigh` / `n` | 管理 ARP 表 |
| `netns` | 管理网络命名空间 |

**示例**：

```bash
# 显示所有接口的 IP 地址
ip addr
ip a

# 显示指定接口
ip addr show eth0

# 添加 IP 地址
sudo ip addr add 192.168.1.100/24 dev eth0

# 删除 IP 地址
sudo ip addr del 192.168.1.100/24 dev eth0

# 启用/禁用接口
sudo ip link set eth0 up
sudo ip link set eth0 down

# 显示路由表
ip route
ip r

# 添加路由
sudo ip route add 10.0.0.0/8 via 192.168.1.1

# 删除路由
sudo ip route del 10.0.0.0/8

# 显示 ARP 表
ip neigh
ip n

# 添加静态 ARP
sudo ip neigh add 192.168.1.1 lladdr 00:11:22:33:44:55 dev eth0
```

---

### 4.7 ifconfig -- 网络接口配置

**说明**：配置和显示网络接口信息（较旧，推荐使用 ip）。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `接口名` | 显示指定接口信息 |
| `up` / `down` | 启用/禁用接口 |
| `inet IP` | 设置 IP 地址 |
| `netmask 掩码` | 设置子网掩码 |
| `broadcast 地址` | 设置广播地址 |
| `hw ether MAC` | 设置 MAC 地址 |
| `-a` | 显示所有接口（包括禁用） |

**示例**：

```bash
# 显示所有活动接口
ifconfig

# 显示指定接口
ifconfig eth0

# 显示所有接口
ifconfig -a

# 启用接口
sudo ifconfig eth0 up

# 禁用接口
sudo ifconfig eth0 down

# 设置 IP 地址
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0

# 设置 MAC 地址
sudo ifconfig eth0 hw ether 00:11:22:33:44:55
```

---

### 4.8 ssh -- 安全远程登录

**说明**：通过加密通道远程登录到另一台主机。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-p 端口` | 指定端口 |
| `-l 用户` | 指定登录用户 |
| `-i 密钥` | 指定私钥文件 |
| `-v` / `-vv` / `-vvv` | 详细模式（调试用） |
| `-N` | 不执行远程命令（用于端口转发） |
| `-f` | 后台运行 |
| `-L 本地:远程:主机` | 本地端口转发 |
| `-R 远程:本地:主机` | 远程端口转发 |
| `-D 端口` | SOCKS 代理 |
| `-C` | 启用压缩 |
| `-o 选项=值` | 指定配置选项 |
| `-t` | 强制分配伪终端 |

**示例**：

```bash
# 基本登录
ssh user@remote-host

# 指定端口
ssh -p 2222 user@remote-host

# 使用密钥登录
ssh -i ~/.ssh/id_rsa user@remote-host

# 执行远程命令
ssh user@remote-host "ls -la"

# 本地端口转发（将本地 8080 转发到远程的 80）
ssh -L 8080:localhost:80 user@remote-host

# 远程端口转发（将远程 8080 转发到本地的 80）
ssh -R 8080:localhost:80 user@remote-host

# SOCKS 代理
ssh -D 1080 user@remote-host

# 后台运行端口转发
ssh -fN -L 3306:db-server:3306 user@remote-host

# 使用配置文件中的别名
ssh myserver

# 复制公钥到远程主机（免密登录）
ssh-copy-id user@remote-host
```

---

### 4.9 scp -- 安全文件复制

**说明**：通过 SSH 安全地复制文件。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-P 端口` | 指定 SSH 端口 |
| `-r` | 递归复制目录 |
| `-p` | 保留文件属性 |
| `-C` | 启用压缩 |
| `-i 密钥` | 指定私钥 |
| `-l 限速` | 限制带宽（Kbit/s） |
| `-v` | 详细模式 |
| `-q` | 静默模式 |

**示例**：

```bash
# 本地复制到远程
scp file.txt user@remote:/path/to/dest/

# 远程复制到本地
scp user@remote:/path/to/file.txt ./

# 复制目录
scp -r mydir/ user@remote:/path/to/dest/

# 指定端口
scp -P 2222 file.txt user@remote:/path/

# 保留属性
scp -p file.txt user@remote:/path/

# 复制多个文件
scp file1.txt file2.txt user@remote:/path/

# 使用密钥
scp -i ~/.ssh/id_rsa file.txt user@remote:/path/

# 两个远程主机之间复制
scp user1@host1:/file.txt user2@host2:/path/
```

---

### 4.10 rsync -- 远程同步工具

**说明**：高效的文件同步和传输工具，支持增量传输。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-a` | 归档模式（递归、保留属性） |
| `-v` | 详细输出 |
| `-z` | 传输时压缩 |
| `-P` | 显示进度，支持断点续传 |
| `--progress` | 显示传输进度 |
| `--delete` | 删除目标中源没有的文件 |
| `-n` / `--dry-run` | 模拟运行（不实际执行） |
| `-e` | 指定远程 shell |
| `--exclude=模式` | 排除文件 |
| `--include=模式` | 包含文件 |
| `-h` | 人类可读格式 |
| `--bwlimit=KBPS` | 限制带宽 |

**示例**：

```bash
# 本地同步目录
rsync -av /source/dir/ /dest/dir/

# 同步到远程（注意末尾斜杠的区别）
rsync -av /local/dir/ user@remote:/remote/dir/

# 从远程同步到本地
rsync -av user@remote:/remote/dir/ /local/dir/

# 显示进度并压缩
rsync -avzP /local/dir/ user@remote:/remote/dir/

# 删除目标中多余的文件（完全镜像）
rsync -av --delete /source/ /dest/

# 排除某些文件
rsync -av --exclude="*.log" --exclude="tmp/" /source/ /dest/

# 模拟运行
rsync -avn /source/ /dest/

# 限制带宽（100KB/s）
rsync -av --bwlimit=100 /source/ user@remote:/dest/

# 使用指定端口 SSH
rsync -av -e "ssh -p 2222" /source/ user@remote:/dest/

# 只同步更新的文件
rsync -avzu /source/ /dest/
```


---

## 5. 压缩归档

### 5.1 tar -- 归档工具

**说明**：将多个文件打包成一个归档文件，常与压缩工具配合使用。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-c` | 创建新归档 |
| `-x` | 提取归档 |
| `-t` | 列出归档内容 |
| `-v` | 详细输出 |
| `-f 文件` | 指定归档文件名 |
| `-z` | 使用 gzip 压缩/解压 |
| `-j` | 使用 bzip2 压缩/解压 |
| `-J` | 使用 xz 压缩/解压 |
| `-C 目录` | 切换到指定目录 |
| `-p` | 保留文件权限 |
| `--exclude=模式` | 排除文件 |
| `--remove-files` | 归档后删除源文件 |

**示例**：

```bash
# 创建 tar 归档
tar -cvf archive.tar file1.txt file2.txt dir/

# 创建 gzip 压缩的 tar 包
tar -czvf archive.tar.gz dir/

# 创建 bzip2 压缩的 tar 包
tar -cjvf archive.tar.bz2 dir/

# 创建 xz 压缩的 tar 包
tar -cJvf archive.tar.xz dir/

# 提取 tar 归档
tar -xvf archive.tar

# 提取 gzip 压缩的 tar 包
tar -xzvf archive.tar.gz

# 提取到指定目录
tar -xzvf archive.tar.gz -C /path/to/dest/

# 列出归档内容
tar -tvf archive.tar.gz

# 排除某些文件
tar -czvf backup.tar.gz --exclude="*.log" --exclude="tmp/" /home/aero/

# 只提取归档中的特定文件
tar -xzvf archive.tar.gz path/inside/file.txt

# 追加文件到归档
tar -rvf archive.tar newfile.txt
```

---

### 5.2 gzip / gunzip -- gzip 压缩

**说明**：使用 gzip 算法压缩/解压文件。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-d` | 解压 |
| `-k` | 保留源文件 |
| `-v` | 详细输出 |
| `-r` | 递归压缩目录 |
| `-1` ~ `-9` | 压缩级别（1 最快，9 最好） |
| `-l` | 列出压缩文件信息 |
| `-t` | 测试压缩文件完整性 |

**示例**：

```bash
# 压缩文件（源文件会被删除）
gzip file.txt

# 压缩并保留源文件
gzip -k file.txt

# 指定压缩级别
gzip -9 file.txt

# 解压文件
gunzip file.txt.gz
# 或
gzip -d file.txt.gz

# 解压并保留压缩文件
gunzip -k file.txt.gz

# 递归压缩目录
gzip -r dir/

# 查看压缩文件信息
gzip -l file.txt.gz

# 测试压缩文件
gzip -t file.txt.gz

# 压缩标准输入
cat file.txt | gzip > file.txt.gz
```

---

### 5.3 zip / unzip -- zip 格式压缩

**说明**：创建和解压 zip 格式的压缩文件。

**zip 常用选项**：

| 选项 | 说明 |
|------|------|
| `-r` | 递归压缩目录 |
| `-e` | 加密压缩 |
| `-q` | 静默模式 |
| `-v` | 详细模式 |
| `-u` | 更新压缩包中的文件 |
| `-d` | 从压缩包中删除文件 |
| `-m` | 压缩后删除源文件 |
| `-9` | 最大压缩 |
| `-x 文件` | 排除文件 |

**unzip 常用选项**：

| 选项 | 说明 |
|------|------|
| `-l` | 列出内容 |
| `-d 目录` | 解压到指定目录 |
| `-o` | 覆盖现有文件 |
| `-n` | 不覆盖现有文件 |
| `-q` | 静默模式 |
| `-v` | 详细模式 |
| `-j` | 不保留目录结构 |
| `-P 密码` | 指定密码 |

**示例**：

```bash
# 压缩文件
zip archive.zip file1.txt file2.txt

# 递归压缩目录
zip -r archive.zip dir/

# 加密压缩
zip -e archive.zip file.txt

# 排除文件
zip -r archive.zip dir/ -x "*.log"

# 更新压缩包
zip -u archive.zip newfile.txt

# 列出压缩包内容
unzip -l archive.zip

# 解压到当前目录
unzip archive.zip

# 解压到指定目录
unzip archive.zip -d /path/to/dest/

# 覆盖现有文件
unzip -o archive.zip

# 不覆盖现有文件
unzip -n archive.zip

# 解压指定文件
unzip archive.zip file.txt

# 解压时指定密码
unzip -P password archive.zip
```

---

## 6. 磁盘管理

### 6.1 fdisk -- 磁盘分区工具

**说明**：管理磁盘分区表。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-l` | 列出所有分区表 |
| `-u` | 使用扇区而非柱面显示 |
| `-s 分区` | 显示分区大小（块） |

**交互命令**：

| 命令 | 说明 |
|------|------|
| `m` | 显示帮助 |
| `p` | 显示分区表 |
| `n` | 新建分区 |
| `d` | 删除分区 |
| `t` | 更改分区类型 |
| `w` | 写入并退出 |
| `q` | 不保存退出 |
| `l` | 列出分区类型 |

**示例**：

```bash
# 列出所有磁盘分区
sudo fdisk -l

# 查看指定磁盘
sudo fdisk -l /dev/sda

# 编辑分区表
sudo fdisk /dev/sdb

# 使用 GPT 分区（gdisk）
sudo gdisk /dev/sdb
```

---

### 6.2 mount -- 挂载文件系统

**说明**：将文件系统挂载到指定挂载点。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-t 类型` | 指定文件系统类型 |
| `-o 选项` | 指定挂载选项 |
| `-a` | 挂载 /etc/fstab 中定义的所有文件系统 |
| `-r` | 只读挂载 |
| `-w` | 读写挂载 |
| `-B` | 绑定挂载 |
| `--bind` | 绑定挂载 |

**常用挂载选项**：

| 选项 | 说明 |
|------|------|
| `ro` / `rw` | 只读 / 读写 |
| `noexec` | 禁止执行文件 |
| `nosuid` | 忽略 SUID/SGID 位 |
| `nodev` | 不解释设备文件 |
| `sync` / `async` | 同步 / 异步写入 |
| `auto` / `noauto` | 自动挂载 |
| `user` / `nouser` | 允许普通用户挂载 |

**示例**：

```bash
# 基本挂载
sudo mount /dev/sdb1 /mnt

# 指定文件系统类型
sudo mount -t ext4 /dev/sdb1 /mnt

# 指定挂载选项
sudo mount -o ro,noexec /dev/sdb1 /mnt

# 挂载 ISO 文件
sudo mount -o loop file.iso /mnt/iso

# 挂载 NFS
sudo mount -t nfs server:/share /mnt/nfs

# 挂载 CIFS/SMB
sudo mount -t cifs //server/share /mnt/smb -o username=user

# 绑定挂载
sudo mount --bind /source/dir /dest/dir

# 挂载所有在 fstab 中定义的文件系统
sudo mount -a

# 查看已挂载的文件系统
mount

# 查看指定挂载点
mount | grep /mnt
```

---

### 6.3 umount -- 卸载文件系统

**说明**：卸载已挂载的文件系统。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-f` | 强制卸载 |
| `-l` | 延迟卸载（lazy） |
| `-r` | 卸载失败时以只读重新挂载 |
| `-a` | 卸载所有文件系统 |
| `-t 类型` | 卸载指定类型的文件系统 |

**示例**：

```bash
# 通过挂载点卸载
sudo umount /mnt

# 通过设备卸载
sudo umount /dev/sdb1

# 强制卸载
sudo umount -f /mnt

# 延迟卸载（当设备正忙时）
sudo umount -l /mnt

# 卸载所有挂载在 /mnt 下的文件系统
sudo umount -a -t nfs
```

---

### 6.4 df / du（已在系统管理中介绍）

参考 [3.4 df](#34-df----查看磁盘空间) 和 [3.5 du](#35-du----查看目录文件大小)。

---

## 7. 软件包管理

### 7.1 apt（Debian/Ubuntu）

**说明**：Advanced Package Tool，Debian 系发行版的包管理工具。

**常用命令**：

| 命令 | 说明 |
|------|------|
| `update` | 更新包列表 |
| `upgrade` | 升级已安装的包 |
| `full-upgrade` | 完整升级（可能删除包） |
| `install 包名` | 安装包 |
| `remove 包名` | 删除包（保留配置） |
| `purge 包名` | 彻底删除包（包括配置） |
| `autoremove` | 删除不再需要的依赖 |
| `search 关键词` | 搜索包 |
| `show 包名` | 显示包信息 |
| `list` | 列出包 |
| `policy 包名` | 显示包的版本策略 |

**示例**：

```bash
# 更新包列表
sudo apt update

# 升级所有包
sudo apt upgrade

# 完整升级
sudo apt full-upgrade

# 安装包
sudo apt install nginx

# 安装多个包
sudo apt install nginx php mysql-server

# 删除包
sudo apt remove nginx

# 彻底删除
sudo apt purge nginx

# 删除不再需要的依赖
sudo apt autoremove

# 清理下载的包缓存
sudo apt clean

# 搜索包
apt search nginx

# 显示包信息
apt show nginx

# 列出已安装的包
apt list --installed

# 查看可更新的包
apt list --upgradable
```

---

### 7.2 yum（CentOS/RHEL 7 及更早）

**说明**：Yellowdog Updater Modified，RHEL/CentOS 系的包管理工具。

**常用命令**：

| 命令 | 说明 |
|------|------|
| `install 包名` | 安装包 |
| `remove 包名` | 删除包 |
| `update` | 更新所有包 |
| `update 包名` | 更新指定包 |
| `search 关键词` | 搜索包 |
| `info 包名` | 显示包信息 |
| `list` | 列出包 |
| `clean all` | 清理缓存 |
| `repolist` | 列出仓库 |

**示例**：

```bash
# 安装包
sudo yum install nginx

# 删除包
sudo yum remove nginx

# 更新所有包
sudo yum update

# 更新指定包
sudo yum update nginx

# 搜索包
yum search nginx

# 显示包信息
yum info nginx

# 列出已安装的包
yum list installed

# 列出可用的包
yum list available

# 清理缓存
sudo yum clean all

# 列出仓库
yum repolist
```

---

### 7.3 dnf（CentOS/RHEL 8+, Fedora）

**说明**：Dandified YUM，yum 的下一代版本，更快更强大。

**常用命令**：

| 命令 | 说明 |
|------|------|
| `install 包名` | 安装包 |
| `remove 包名` | 删除包 |
| `upgrade` | 升级所有包 |
| `search 关键词` | 搜索包 |
| `info 包名` | 显示包信息 |
| `history` | 查看操作历史 |
| `autoremove` | 删除不再需要的依赖 |

**示例**：

```bash
# 安装包
sudo dnf install nginx

# 删除包
sudo dnf remove nginx

# 升级所有包
sudo dnf upgrade

# 搜索包
dnf search nginx

# 显示包信息
dnf info nginx

# 查看操作历史
dnf history

# 撤销操作
dnf history undo last

# 自动删除依赖
sudo dnf autoremove

# 清理缓存
sudo dnf clean all
```

---

### 7.4 pacman（Arch Linux）

**说明**：Arch Linux 的包管理工具。

**常用命令**：

| 命令 | 说明 |
|------|------|
| `-S 包名` | 安装包 |
| `-R 包名` | 删除包 |
| `-Rs 包名` | 删除包及其依赖 |
| `-Rns 包名` | 彻底删除（包括配置） |
| `-Syu` | 同步并升级系统 |
| `-Ss 关键词` | 搜索包 |
| `-Si 包名` | 显示包信息 |
| `-Q` | 列出已安装的包 |
| `-Qe` | 列出显式安装的包 |
| `-Sc` | 清理缓存 |

**示例**：

```bash
# 同步包数据库
sudo pacman -Sy

# 升级系统
sudo pacman -Syu

# 安装包
sudo pacman -S nginx

# 删除包
sudo pacman -R nginx

# 删除包及未使用的依赖
sudo pacman -Rs nginx

# 搜索包
pacman -Ss nginx

# 显示包信息
pacman -Si nginx

# 列出已安装的包
pacman -Q

# 列出显式安装的包
pacman -Qe

# 清理缓存
sudo pacman -Sc

# 彻底清理缓存
sudo pacman -Scc
```


---

## 8. 其他实用命令

### 8.1 history -- 查看命令历史

**说明**：显示已执行过的命令历史。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-c` | 清除历史 |
| `-d N` | 删除第 N 条历史 |
| `N` | 显示最近 N 条 |

**快捷操作**：

| 操作 | 说明 |
|------|------|
| `!N` | 执行第 N 条历史命令 |
| `!-N` | 执行倒数第 N 条 |
| `!!` | 执行上一条命令 |
| `!字符串` | 执行最近以该字符串开头的命令 |
| `!?字符串?` | 执行最近包含该字符串的命令 |
| `^old^new` | 替换上条命令中的字符串并执行 |
| `Ctrl+R` | 反向搜索历史 |

**示例**：

```bash
# 显示历史
history

# 显示最近 20 条
history 20

# 执行第 100 条历史命令
!100

# 执行上一条命令
!!

# 执行最近以 sudo 开头的命令
!sudo

# 替换上条命令中的字符串
^old^new

# 清除历史
history -c

# 搜索历史
history | grep "apt"
```

---

### 8.2 alias -- 命令别名

**说明**：为命令创建别名，简化输入。

**常用操作**：

| 操作 | 说明 |
|------|------|
| `alias` | 列出所有别名 |
| `alias 名称='命令'` | 创建别名 |
| `unalias 名称` | 删除别名 |
| `type 名称` | 查看命令类型 |

**示例**：

```bash
# 列出所有别名
alias

# 创建别名
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias ..='cd ..'
alias ...='cd ../..'
alias update='sudo apt update && sudo apt upgrade'

# 删除别名
unalias ll

# 查看命令类型
type ll
type ls

# 永久别名（添加到 ~/.bashrc 或 ~/.zshrc）
echo "alias ll='ls -alF'" >> ~/.bashrc
source ~/.bashrc
```

---

### 8.3 echo -- 输出文本

**说明**：输出字符串或变量值。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-n` | 不换行 |
| `-e` | 启用转义字符 |
| `-E` | 禁用转义字符 |

**转义字符**：

| 字符 | 说明 |
|------|------|
| `\n` | 换行 |
| `\t` | 制表符 |
| `\\` | 反斜杠 |
| `\a` | 响铃 |
| `\b` | 退格 |

**示例**：

```bash
# 基本输出
echo "Hello, World!"

# 输出变量
name="Linux"
echo "Hello, $name!"

# 不换行输出
echo -n "Loading..."

# 使用转义字符
echo -e "Line 1\nLine 2\tTabbed"

# 输出到文件
echo "Hello" > file.txt

# 追加到文件
echo "World" >> file.txt

# 输出命令结果
echo "Today is $(date)"

# 输出环境变量
echo $PATH
echo $HOME
echo $USER
```

---

### 8.4 date -- 日期时间

**说明**：显示或设置系统日期和时间。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `+格式` | 自定义输出格式 |
| `-d "字符串"` | 解析指定日期 |
| `-s "字符串"` | 设置日期时间 |
| `-u` | 显示 UTC 时间 |
| `-R` | 显示 RFC 2822 格式 |
| `-I` | 显示 ISO 8601 格式 |

**常用格式**：

| 格式 | 说明 |
|------|------|
| `%Y` | 四位年份 |
| `%m` | 月份 (01-12) |
| `%d` | 日期 (01-31) |
| `%H` | 小时 (00-23) |
| `%M` | 分钟 (00-59) |
| `%S` | 秒 (00-59) |
| `%s` | Unix 时间戳 |
| `%A` | 星期全称 |
| `%B` | 月份全称 |
| `%Z` | 时区 |

**示例**：

```bash
# 显示当前日期时间
date

# 自定义格式
date +"%Y-%m-%d %H:%M:%S"

# 只显示日期
date +"%Y-%m-%d"

# 只显示时间
date +"%H:%M:%S"

# 显示 Unix 时间戳
date +%s

# 解析指定日期
date -d "2024-01-01"
date -d "yesterday"
date -d "next Monday"
date -d "3 days ago"

# 时间戳转日期
date -d "@1700000000"

# ISO 格式
date -I
date -Iseconds

# RFC 格式
date -R

# 设置日期时间（需要 root）
sudo date -s "2024-01-01 12:00:00"
```

---

### 8.5 which -- 查找命令位置

**说明**：查找可执行文件的路径。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-a` | 显示所有匹配的路径 |

**示例**：

```bash
# 查找命令位置
which python
which nginx
which git

# 显示所有匹配
which -a python

# 检查命令是否存在
which nginx && echo "nginx installed" || echo "nginx not found"
```

---

### 8.6 whereis -- 查找程序位置

**说明**：查找程序的二进制文件、源代码和手册页位置。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-b` | 只查找二进制文件 |
| `-m` | 只查找手册页 |
| `-s` | 只查找源代码 |
| `-B 目录` | 指定搜索二进制文件的目录 |
| `-M 目录` | 指定搜索手册页的目录 |

**示例**：

```bash
# 查找程序的所有相关文件
whereis python
whereis nginx

# 只查找二进制文件
whereis -b python

# 只查找手册页
whereis -m python
```

---

### 8.7 man -- 查看手册页

**说明**：查看命令的详细手册页。

**常用选项**：

| 选项 | 说明 |
|------|------|
| `-k 关键词` | 搜索手册页 |
| `-f 命令` | 显示命令的简短描述 |
| `-a` | 显示所有匹配的手册页 |
| `-N` | 不格式化输出 |

**手册页章节**：

| 章节 | 内容 |
|------|------|
| 1 | 用户命令 |
| 2 | 系统调用 |
| 3 | C 库函数 |
| 4 | 设备和特殊文件 |
| 5 | 文件格式和约定 |
| 6 | 游戏 |
| 7 | 杂项 |
| 8 | 系统管理命令 |

**示例**：

```bash
# 查看命令手册
man ls
man ssh

# 查看指定章节
man 5 passwd      # 查看 passwd 文件格式
man 3 printf      # 查看 C 库函数 printf

# 搜索手册页
man -k network
man -k "copy file"

# 显示简短描述
man -f ls
whatis ls         # 等同于 man -f

# 查看所有匹配的手册页
man -a printf
```

---

### 8.8 info -- 查看 info 文档

**说明**：查看 GNU 项目的 info 格式文档（通常比 man 更详细）。

**常用操作**：

| 按键 | 说明 |
|------|------|
| `?` | 显示帮助 |
| `Space` / `PageDown` | 向下翻页 |
| `PageUp` / `b` | 向上翻页 |
| `n` | 下一节 |
| `p` | 上一节 |
| `u` | 上一级 |
| `q` | 退出 |
| `Enter` | 跟随链接 |
| `l` | 返回上一页 |
| `d` | 返回目录 |
| `s` | 搜索 |
| `/` | 正则搜索 |

**示例**：

```bash
# 查看 info 文档
info ls
info bash
info coreutils

# 显示目录
info

# 搜索主题
info --apropos "copy file"
```

---

## 9. 命令速查表

### 9.1 文件和目录操作速查

| 命令 | 常用用法 |
|------|----------|
| `ls` | `ls -lah`, `ls -lt`, `ls -R` |
| `cd` | `cd ~`, `cd -`, `cd ..` |
| `pwd` | `pwd`, `pwd -P` |
| `mkdir` | `mkdir -p`, `mkdir -m 755` |
| `rm` | `rm -rf`, `rm -i`, `rm -v` |
| `cp` | `cp -r`, `cp -a`, `cp -v` |
| `mv` | `mv -i`, `mv -v` |
| `touch` | `touch file`, `touch -t 202401011200` |
| `cat` | `cat file`, `cat -n`, `cat > file` |
| `less` | `less file`, `less +F`, `less -N` |
| `head` | `head -n 20`, `head -c 100` |
| `tail` | `tail -f`, `tail -n 50` |
| `find` | `find . -name "*.txt"`, `find . -size +100M` |
| `chmod` | `chmod 755`, `chmod u+x`, `chmod -R` |
| `chown` | `chown user:group`, `chown -R` |
| `ln` | `ln -s`, `ln -sf` |

### 9.2 文本处理速查

| 命令 | 常用用法 |
|------|----------|
| `grep` | `grep -i`, `grep -r`, `grep -n`, `grep -C 3` |
| `sed` | `sed 's/old/new/g'`, `sed -i`, `sed -n '5p'` |
| `awk` | `awk '{print $1}'`, `awk -F: '{print $1}'` |
| `sort` | `sort -n`, `sort -k2`, `sort -u` |
| `uniq` | `uniq -c`, `uniq -d`, `sort \| uniq` |
| `cut` | `cut -d: -f1`, `cut -c1-10` |
| `tr` | `tr 'a-z' 'A-Z'`, `tr -d`, `tr -s` |
| `wc` | `wc -l`, `wc -w`, `wc -c` |
| `diff` | `diff -u`, `diff -r`, `diff -y` |

### 9.3 系统管理速查

| 命令 | 常用用法 |
|------|----------|
| `ps` | `ps aux`, `ps -ef`, `ps aux --sort=-%cpu` |
| `top` | `top`, `top -p PID`, `top -u user` |
| `kill` | `kill PID`, `kill -9 PID`, `killall name` |
| `df` | `df -h`, `df -Th`, `df -i` |
| `du` | `du -sh`, `du -h --max-depth=1` |
| `free` | `free -h`, `free -m` |
| `uname` | `uname -a`, `uname -r`, `uname -m` |
| `uptime` | `uptime` |
| `who` | `who`, `w`, `last` |
| `crontab` | `crontab -e`, `crontab -l` |
| `systemctl` | `start`, `stop`, `restart`, `enable`, `status` |

### 9.4 网络管理速查

| 命令 | 常用用法 |
|------|----------|
| `ping` | `ping -c 4 host`, `ping -i 2 host` |
| `curl` | `curl -o file URL`, `curl -I URL`, `curl -X POST` |
| `wget` | `wget URL`, `wget -c URL`, `wget -O file URL` |
| `netstat` | `netstat -tlnp`, `netstat -rn` |
| `ss` | `ss -tlnp`, `ss -s` |
| `ip` | `ip addr`, `ip route`, `ip link` |
| `ifconfig` | `ifconfig`, `ifconfig eth0 up` |
| `ssh` | `ssh user@host`, `ssh -p 2222`, `ssh -L` |
| `scp` | `scp file user@host:/path`, `scp -r` |
| `rsync` | `rsync -avzP source/ dest/` |

### 9.5 压缩归档速查

| 命令 | 常用用法 |
|------|----------|
| `tar` | `tar -czvf`, `tar -xzvf`, `tar -tvf` |
| `gzip` | `gzip file`, `gunzip file.gz`, `gzip -k` |
| `zip` | `zip -r archive.zip dir/`, `unzip archive.zip` |

### 9.6 磁盘管理速查

| 命令 | 常用用法 |
|------|----------|
| `fdisk` | `fdisk -l`, `fdisk /dev/sdb` |
| `mount` | `mount /dev/sdb1 /mnt`, `mount -o loop` |
| `umount` | `umount /mnt`, `umount -f` |

### 9.7 软件包管理速查

| 系统 | 安装 | 删除 | 更新 | 搜索 |
|------|------|------|------|------|
| Debian/Ubuntu | `apt install` | `apt remove` | `apt upgrade` | `apt search` |
| CentOS/RHEL 7 | `yum install` | `yum remove` | `yum update` | `yum search` |
| CentOS/RHEL 8+ | `dnf install` | `dnf remove` | `dnf upgrade` | `dnf search` |
| Arch Linux | `pacman -S` | `pacman -R` | `pacman -Syu` | `pacman -Ss` |

### 9.8 其他实用命令速查

| 命令 | 常用用法 |
|------|----------|
| `history` | `history`, `!N`, `!!`, `Ctrl+R` |
| `alias` | `alias`, `alias ll='ls -alF'` |
| `echo` | `echo "text"`, `echo $VAR`, `echo -e` |
| `date` | `date`, `date +"%Y-%m-%d"`, `date -d "yesterday"` |
| `which` | `which command`, `which -a` |
| `whereis` | `whereis command`, `whereis -b` |
| `man` | `man command`, `man -k keyword` |
| `info` | `info command` |

---

## 附录：常用快捷键

### Bash 快捷键

| 快捷键 | 说明 |
|--------|------|
| `Ctrl+A` | 移到行首 |
| `Ctrl+E` | 移到行尾 |
| `Ctrl+U` | 删除光标前所有内容 |
| `Ctrl+K` | 删除光标后所有内容 |
| `Ctrl+W` | 删除光标前一个单词 |
| `Ctrl+Y` | 粘贴删除的内容 |
| `Ctrl+L` | 清屏 |
| `Ctrl+C` | 中断当前命令 |
| `Ctrl+D` | 退出/EOF |
| `Ctrl+Z` | 挂起当前进程 |
| `Ctrl+R` | 反向搜索历史 |
| `Ctrl+G` | 取消搜索 |
| `Tab` | 自动补全 |
| `Tab Tab` | 显示所有补全选项 |
| `Alt+.` | 插入上条命令的最后一个参数 |
| `Ctrl+P` / `↑` | 上一条命令 |
| `Ctrl+N` / `↓` | 下一条命令 |

### Vim 基础快捷键

| 快捷键 | 说明 |
|--------|------|
| `i` | 进入插入模式 |
| `Esc` | 返回普通模式 |
| `:w` | 保存 |
| `:q` | 退出 |
| `:wq` / `:x` | 保存并退出 |
| `:q!` | 强制退出不保存 |
| `dd` | 删除一行 |
| `yy` | 复制一行 |
| `p` | 粘贴 |
| `u` | 撤销 |
| `Ctrl+R` | 重做 |
| `gg` | 跳到文件开头 |
| `G` | 跳到文件末尾 |
| `/pattern` | 搜索 |
| `n` | 下一个匹配 |
| `N` | 上一个匹配 |

---

## 附录：权限数字对照表

| 数字 | 权限 | 说明 |
|------|------|------|
| 7 | `rwx` | 读、写、执行 |
| 6 | `rw-` | 读、写 |
| 5 | `r-x` | 读、执行 |
| 4 | `r--` | 只读 |
| 3 | `-wx` | 写、执行 |
| 2 | `-w-` | 只写 |
| 1 | `--x` | 只执行 |
| 0 | `---` | 无权限 |

**常用权限组合**：

| 权限 | 适用场景 |
|------|----------|
| `777` | 所有人完全控制（慎用） |
| `755` | 所有者完全控制，其他人读+执行（目录默认） |
| `750` | 所有者完全控制，组读+执行，其他人无权限 |
| `700` | 只有所有者有权限（私有文件） |
| `644` | 所有者读写，其他人只读（文件默认） |
| `640` | 所有者读写，组只读，其他人无权限 |
| `600` | 只有所有者可读写（密钥文件） |
| `666` | 所有人可读写 |

---

## 附录：特殊文件描述符

| 描述符 | 名称 | 说明 |
|--------|------|------|
| `0` | stdin | 标准输入 |
| `1` | stdout | 标准输出 |
| `2` | stderr | 标准错误 |

**重定向操作**：

| 操作 | 说明 |
|------|------|
| `> file` | 标准输出重定向到文件（覆盖） |
| `>> file` | 标准输出重定向到文件（追加） |
| `2> file` | 标准错误重定向到文件 |
| `2>&1` | 标准错误重定向到标准输出 |
| `>> file 2>&1` | 标准输出和标准错误都追加到文件 |
| `< file` | 从文件读取标准输入 |
| `<< EOF` | Here Document |
| `<<< "string"` | Here String |
| `> /dev/null` | 丢弃输出 |
| `2> /dev/null` | 丢弃错误 |
| `> /dev/null 2>&1` | 丢弃所有输出 |

---

> **文档版本**：v1.0  
> **最后更新**：2024年  
> **适用系统**：Linux (CentOS/Ubuntu/Debian/Arch 等)
