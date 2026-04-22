#!/usr/bin/env python3
import json
import os
import re

# 读取 wiki.json
with open('2026-04-22-120026/wiki.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按 group 组织
sections = {}
for page in data['pages']:
    group = page.get('group', page.get('section', '其他'))
    if group not in sections:
        sections[group] = []
    sections[group].append(page)

# 读取所有 md 文件内容
contents = {}
for page in data['pages']:
    filepath = '2026-04-22-120026/' + page['file']
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            contents[page['slug']] = f.read()
    else:
        contents[page['slug']] = f'文件未找到: {filepath}'

# Markdown 简单转换函数
def md_to_html(md):
    # 代码块 (先处理，避免内部被转换)
    code_blocks = []
    def save_code(m):
        lang = m.group(1) or ''
        code = m.group(2)
        if lang == 'mermaid':
            code_blocks.append(f'<div class="mermaid">{code}</div>')
        else:
            code_blocks.append(f'<pre><code>{code}</code></pre>')
        return f'__CODE_BLOCK_{len(code_blocks)-1}__'
    md = re.sub(r'```(\w+)?\n(.*?)```', save_code, md, flags=re.DOTALL)
    
    # 转义 HTML (排除已处理的代码块占位符)
    def escape_html(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # 分段处理：保护占位符不被转义
    parts = []
    last_end = 0
    for m in re.finditer(r'__CODE_BLOCK_\d+__', md):
        parts.append(escape_html(md[last_end:m.start()]))
        parts.append(m.group())
        last_end = m.end()
    parts.append(escape_html(md[last_end:]))
    md = ''.join(parts)
    
    # 行内代码
    md = re.sub(r'`([^`]+)`', r'<code>\1</code>', md)
    
    # 标题
    md = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', md, flags=re.MULTILINE)
    md = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', md, flags=re.MULTILINE)
    md = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', md, flags=re.MULTILINE)
    md = re.sub(r'^### (.+)$', r'<h3>\1</h3>', md, flags=re.MULTILINE)
    md = re.sub(r'^## (.+)$', r'<h2>\1</h2>', md, flags=re.MULTILINE)
    md = re.sub(r'^# (.+)$', r'<h1>\1</h1>', md, flags=re.MULTILINE)
    
    # 粗体、斜体
    md = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', md)
    md = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', md)
    md = re.sub(r'\*(.+?)\*', r'<em>\1</em>', md)
    
    # 引用
    md = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', md, flags=re.MULTILINE)
    
    # 链接 [text](url)
    md = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', md)
    
    # 图片
    md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;">', md)
    
    # 水平线
    md = re.sub(r'^---+$', r'<hr>', md, flags=re.MULTILINE)
    
    # 表格处理
    lines = md.split('\n')
    result = []
    in_table = False
    table_rows = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            if not in_table:
                in_table = True
                table_rows = [stripped]
            else:
                table_rows.append(stripped)
        elif in_table:
            if len(table_rows) >= 2:
                html_table = '<table>\n'
                for i, row in enumerate(table_rows):
                    if i == 1 and set(row.strip()) <= set('|-: '):
                        continue
                    cells = [c.strip() for c in row.split('|')[1:-1]]
                    tag = 'th' if i == 0 else 'td'
                    html_table += '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>\n'
                html_table += '</table>'
                result.append(html_table)
            else:
                result.extend(table_rows)
            in_table = False
            result.append(line)
        else:
            result.append(line)
    
    if in_table and len(table_rows) >= 2:
        html_table = '<table>\n'
        for i, row in enumerate(table_rows):
            if i == 1 and set(row.strip()) <= set('|-: '):
                continue
            cells = [c.strip() for c in row.split('|')[1:-1]]
            tag = 'th' if i == 0 else 'td'
            html_table += '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>\n'
        html_table += '</table>'
        result.append(html_table)
    elif in_table:
        result.extend(table_rows)
    
    md = '\n'.join(result)
    
    # 无序列表
    md = re.sub(r'^\s*-\s+(.+)$', r'<li>\1</li>', md, flags=re.MULTILINE)
    md = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\g<0></ul>', md, flags=re.DOTALL)
    
    # 有序列表
    md = re.sub(r'^\s*\d+\.\s+(.+)$', r'<li>\1</li>', md, flags=re.MULTILINE)
    md = re.sub(r'(<li>.*?</li>\n)+', r'<ol>\g<0></ol>', md, flags=re.DOTALL)
    
    # 恢复代码块
    for i, block in enumerate(code_blocks):
        md = md.replace(f'__CODE_BLOCK_{i}__', block)
    
    # 段落
    paragraphs = md.split('\n\n')
    new_paras = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith('<') and not p.startswith('<li>'):
            new_paras.append(p)
        else:
            new_paras.append(f'<p>{p}</p>')
    
    return '\n\n'.join(new_paras)

# 生成 HTML
html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorFlow MUSA Extension Wiki</title>
    <style>
        :root {
            --bg: #f5f7fa;
            --sidebar-bg: #1e293b;
            --sidebar-text: #cbd5e1;
            --sidebar-active: #3b82f6;
            --content-bg: #ffffff;
            --text: #334155;
            --heading: #0f172a;
            --border: #e2e8f0;
            --code-bg: #f1f5f9;
            --link: #2563eb;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }
        .container {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 300px;
            background: var(--sidebar-bg);
            color: var(--sidebar-text);
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            padding: 20px 0;
        }
        .sidebar-header {
            padding: 0 20px 20px;
            border-bottom: 1px solid #334155;
        }
        .sidebar-header h1 {
            font-size: 18px;
            color: #fff;
            margin-bottom: 5px;
        }
        .sidebar-header p {
            font-size: 12px;
            color: #94a3b8;
        }
        .nav-group {
            margin: 15px 0;
        }
        .nav-group-title {
            padding: 8px 20px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
        }
        .nav-item {
            display: block;
            padding: 8px 20px 8px 30px;
            color: var(--sidebar-text);
            text-decoration: none;
            font-size: 14px;
            border-left: 3px solid transparent;
            transition: all 0.2s;
        }
        .nav-item:hover {
            background: #334155;
            color: #fff;
        }
        .nav-item.active {
            background: #334155;
            border-left-color: var(--sidebar-active);
            color: #fff;
        }
        .nav-item .level {
            font-size: 10px;
            padding: 1px 6px;
            border-radius: 3px;
            margin-left: 6px;
            background: #475569;
            color: #cbd5e1;
        }
        .main-content {
            margin-left: 300px;
            flex: 1;
            max-width: 900px;
            padding: 40px 50px;
            background: var(--content-bg);
            min-height: 100vh;
        }
        .content-section {
            display: none;
        }
        .content-section.active {
            display: block;
        }
        .content-section h1 {
            font-size: 32px;
            color: var(--heading);
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--border);
        }
        .content-section h2 {
            font-size: 24px;
            color: var(--heading);
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }
        .content-section h3 {
            font-size: 20px;
            color: var(--heading);
            margin: 25px 0 12px;
        }
        .content-section h4 {
            font-size: 17px;
            color: var(--heading);
            margin: 20px 0 10px;
        }
        .content-section p {
            margin: 12px 0;
        }
        .content-section code {
            background: var(--code-bg);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.9em;
            color: #c7254e;
        }
        .content-section pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 16px 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 15px 0;
            font-size: 14px;
            line-height: 1.6;
        }
        .content-section pre code {
            background: transparent;
            color: inherit;
            padding: 0;
        }
        .content-section table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }
        .content-section th,
        .content-section td {
            border: 1px solid var(--border);
            padding: 10px 14px;
            text-align: left;
        }
        .content-section th {
            background: var(--code-bg);
            font-weight: 600;
            color: var(--heading);
        }
        .content-section tr:nth-child(even) {
            background: #f8fafc;
        }
        .content-section ul,
        .content-section ol {
            margin: 12px 0 12px 25px;
        }
        .content-section li {
            margin: 6px 0;
        }
        .content-section blockquote {
            border-left: 4px solid var(--sidebar-active);
            background: #eff6ff;
            padding: 12px 18px;
            margin: 15px 0;
            border-radius: 0 6px 6px 0;
        }
        .content-section a {
            color: var(--link);
            text-decoration: none;
        }
        .content-section a:hover {
            text-decoration: underline;
        }
        .content-section hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 30px 0;
        }
        .mermaid {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin: 15px 0;
            font-family: monospace;
            white-space: pre;
            overflow-x: auto;
        }
        .home-intro {
            text-align: center;
            padding: 60px 20px;
        }
        .home-intro h1 {
            font-size: 42px;
            margin-bottom: 20px;
        }
        .home-intro p {
            font-size: 18px;
            color: #64748b;
            max-width: 600px;
            margin: 0 auto 30px;
        }
        .home-stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-item .number {
            font-size: 36px;
            font-weight: 700;
            color: var(--sidebar-active);
        }
        .stat-item .label {
            font-size: 14px;
            color: #64748b;
            margin-top: 5px;
        }
        @media (max-width: 768px) {
            .sidebar { width: 260px; }
            .main-content { margin-left: 260px; padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <nav class="sidebar">
            <div class="sidebar-header">
                <h1>TF MUSA Extension</h1>
                <p>Wiki Documentation</p>
            </div>
'''

# 生成导航
for group, pages in sections.items():
    html += f'            <div class="nav-group">\n'
    html += f'                <div class="nav-group-title">{group}</div>\n'
    for page in pages:
        level = page.get('level', '')
        level_tag = f'<span class="level">{level}</span>' if level else ''
        html += f'                <a href="#{page["slug"]}" class="nav-item" data-target="{page["slug"]}">{page["title"]}{level_tag}</a>\n'
    html += '            </div>\n'

html += '''        </nav>
        <main class="main-content">
            <div id="home" class="content-section active">
                <div class="home-intro">
                    <h1>TensorFlow MUSA Extension</h1>
                    <p>摩尔线程 MUSA GPU 的 TensorFlow 插件化扩展完整文档。涵盖架构设计、算子实现、图优化、调试与测试等全链路技术细节。</p>
                    <div class="home-stats">
                        <div class="stat-item">
                            <div class="number">22</div>
                            <div class="label">文档章节</div>
                        </div>
                        <div class="stat-item">
                            <div class="number">6</div>
                            <div class="label">核心模块</div>
                        </div>
                        <div class="stat-item">
                            <div class="number">150+</div>
                            <div class="label">算子实现</div>
                        </div>
                    </div>
                </div>
            </div>
'''

# 生成内容区
for page in data['pages']:
    content = contents.get(page['slug'], '')
    html_content = md_to_html(content)
    html += f'''            <div id="{page['slug']}" class="content-section">
                {html_content}
            </div>
'''

html += '''        </main>
    </div>
    <script>
        // 导航切换
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                const target = this.getAttribute('data-target');
                
                // 更新导航状态
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
                
                // 显示对应内容
                document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
                document.getElementById(target).classList.add('active');
                
                // 滚动到顶部
                window.scrollTo(0, 0);
            });
        });
        
        // URL hash 支持
        function handleHash() {
            const hash = window.location.hash.slice(1);
            if (hash) {
                const target = document.querySelector(`[data-target="${hash}"]`);
                if (target) {
                    target.click();
                }
            }
        }
        window.addEventListener('hashchange', handleHash);
        handleHash();
    </script>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: false, theme: 'default' });
        
        // 在内容切换后重新渲染 mermaid
        function renderMermaid() {
            mermaid.run({ querySelector: '.mermaid' });
        }
        
        // 初始渲染
        renderMermaid();
        
        // 监听导航点击，在内容显示后渲染
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', function() {
                setTimeout(renderMermaid, 50);
            });
        });
    </script>
</body>
</html>'''

# 写入文件
with open('wiki.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('✅ HTML 生成成功: wiki.html')
print(f'   文件大小: {os.path.getsize("wiki.html") / 1024:.1f} KB')
