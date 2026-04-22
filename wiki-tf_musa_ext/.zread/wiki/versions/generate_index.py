#!/usr/bin/env python3
"""
Generate index.html that embeds all Markdown content directly in JavaScript.
This avoids CORS issues when opening file:// locally.
"""
import json
import os

# Read wiki.json
with open('2026-04-22-120026/wiki.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Read all markdown files and embed them
pages_data = []
for page in data['pages']:
    filepath = '2026-04-22-120026/' + page['file']
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = f'文件未找到: {filepath}'
    
    # Escape for JavaScript string
    content = content.replace('\\', '\\\\')
    content = content.replace('`', '\\`')
    content = content.replace('$', '\\$')
    
    pages_data.append({
        'slug': page['slug'],
        'title': page['title'],
        'group': page.get('group', page.get('section', '其他')),
        'level': page.get('level', ''),
        'content': content
    })

# Generate navigation HTML
nav_html = ''
current_group = None
for page in pages_data:
    if page['group'] != current_group:
        if current_group is not None:
            nav_html += '            </div>\n'
        nav_html += f'            <div class="nav-group">\n'
        nav_html += f'                <div class="nav-group-title">{page["group"]}</div>\n'
        current_group = page['group']
    
    level_tag = f'<span class="level">{page["level"]}</span>' if page['level'] else ''
    nav_html += f'                <a href="#{page["slug"]}" class="nav-item" data-slug="{page["slug"]}">{page["title"]}{level_tag}</a>\n'

if current_group is not None:
    nav_html += '            </div>\n'

# Generate JS data
js_pages = []
for page in pages_data:
    js_pages.append(f'''    "{page['slug']}": `{page['content']}`''')

js_data = ',\n'.join(js_pages)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorFlow MUSA Extension Wiki</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        :root {{
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
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }}
        .container {{
            display: flex;
            min-height: 100vh;
        }}
        .sidebar {{
            width: 300px;
            background: var(--sidebar-bg);
            color: var(--sidebar-text);
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            padding: 20px 0;
            z-index: 100;
        }}
        .sidebar-header {{
            padding: 0 20px 20px;
            border-bottom: 1px solid #334155;
        }}
        .sidebar-header h1 {{
            font-size: 18px;
            color: #fff;
            margin-bottom: 5px;
        }}
        .sidebar-header p {{
            font-size: 12px;
            color: #94a3b8;
        }}
        .nav-group {{
            margin: 15px 0;
        }}
        .nav-group-title {{
            padding: 8px 20px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            font-weight: 600;
        }}
        .nav-item {{
            display: block;
            padding: 8px 20px 8px 30px;
            color: var(--sidebar-text);
            text-decoration: none;
            font-size: 14px;
            border-left: 3px solid transparent;
            transition: all 0.2s;
            cursor: pointer;
        }}
        .nav-item:hover {{
            background: #334155;
            color: #fff;
        }}
        .nav-item.active {{
            background: #334155;
            border-left-color: var(--sidebar-active);
            color: #fff;
        }}
        .nav-item .level {{
            font-size: 10px;
            padding: 1px 6px;
            border-radius: 3px;
            margin-left: 6px;
            background: #475569;
            color: #cbd5e1;
        }}
        .main-content {{
            margin-left: 300px;
            flex: 1;
            max-width: 900px;
            padding: 40px 50px;
            background: var(--content-bg);
            min-height: 100vh;
        }}
        .content-section {{
            display: none;
        }}
        .content-section.active {{
            display: block;
        }}
        
        /* Markdown content styles */
        .markdown-body h1 {{
            font-size: 32px;
            color: var(--heading);
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--border);
        }}
        .markdown-body h2 {{
            font-size: 24px;
            color: var(--heading);
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}
        .markdown-body h3 {{
            font-size: 20px;
            color: var(--heading);
            margin: 25px 0 12px;
        }}
        .markdown-body h4 {{
            font-size: 17px;
            color: var(--heading);
            margin: 20px 0 10px;
        }}
        .markdown-body p {{
            margin: 12px 0;
        }}
        .markdown-body code {{
            background: var(--code-bg);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.9em;
            color: #c7254e;
        }}
        .markdown-body pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 16px 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 15px 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        .markdown-body pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        .markdown-body table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        .markdown-body th,
        .markdown-body td {{
            border: 1px solid var(--border);
            padding: 10px 14px;
            text-align: left;
        }}
        .markdown-body th {{
            background: var(--code-bg);
            font-weight: 600;
            color: var(--heading);
        }}
        .markdown-body tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .markdown-body ul,
        .markdown-body ol {{
            margin: 12px 0 12px 25px;
        }}
        .markdown-body li {{
            margin: 6px 0;
        }}
        .markdown-body blockquote {{
            border-left: 4px solid var(--sidebar-active);
            background: #eff6ff;
            padding: 12px 18px;
            margin: 15px 0;
            border-radius: 0 6px 6px 0;
        }}
        .markdown-body a {{
            color: var(--link);
            text-decoration: none;
        }}
        .markdown-body a:hover {{
            text-decoration: underline;
        }}
        .markdown-body hr {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 30px 0;
        }}
        .markdown-body img {{
            max-width: 100%;
        }}
        .mermaid {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin: 15px 0;
        }}
        .home-intro {{
            text-align: center;
            padding: 60px 20px;
        }}
        .home-intro h1 {{
            font-size: 42px;
            margin-bottom: 20px;
            color: var(--heading);
        }}
        .home-intro p {{
            font-size: 18px;
            color: #64748b;
            max-width: 600px;
            margin: 0 auto 30px;
        }}
        .home-stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-item .number {{
            font-size: 36px;
            font-weight: 700;
            color: var(--sidebar-active);
        }}
        .stat-item .label {{
            font-size: 14px;
            color: #64748b;
            margin-top: 5px;
        }}
        .loading {{
            text-align: center;
            padding: 40px;
            color: #64748b;
        }}
        @media (max-width: 768px) {{
            .sidebar {{ width: 260px; }}
            .main-content {{ margin-left: 260px; padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <nav class="sidebar">
            <div class="sidebar-header">
                <h1>TF MUSA Extension</h1>
                <p>Wiki Documentation</p>
            </div>
{nav_html}
        </nav>
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
            <div id="content" class="content-section markdown-body"></div>
        </main>
    </div>
    
    <script>
        // Initialize Mermaid
        mermaid.initialize({{
            startOnLoad: false,
            theme: 'default',
            securityLevel: 'loose'
        }});
        
        // Embedded markdown content
        const pages = {{
{js_data}
        }};
        
        // Configure marked to handle mermaid
        const renderer = new marked.Renderer();
        const originalCode = renderer.code.bind(renderer);
        renderer.code = function(code, language) {{
            if (language === 'mermaid') {{
                return '<div class="mermaid">' + code + '</div>';
            }}
            return originalCode(code, language);
        }};
        marked.setOptions({{
            renderer: renderer,
            breaks: true,
            gfm: true
        }});
        
        // Load markdown content
        async function loadMarkdown(slug) {{
            const content = pages[slug];
            if (!content) {{
                document.getElementById('content').innerHTML = 
                    '<div class="loading">内容未找到: ' + slug + '</div>';
                document.getElementById('home').classList.remove('active');
                document.getElementById('content').classList.add('active');
                return;
            }}
            
            // Render markdown
            const contentDiv = document.getElementById('content');
            contentDiv.innerHTML = marked.parse(content);
            
            // Render mermaid diagrams
            await mermaid.run({{
                querySelector: '.mermaid'
            }});
            
            // Show content, hide home
            document.getElementById('home').classList.remove('active');
            contentDiv.classList.add('active');
            
            // Scroll to top
            window.scrollTo(0, 0);
        }}
        
        // Navigation click handler
        document.querySelectorAll('.nav-item').forEach(item => {{
            item.addEventListener('click', function(e) {{
                e.preventDefault();
                const slug = this.getAttribute('data-slug');
                
                // Update nav state
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
                
                // Load content
                loadMarkdown(slug);
                
                // Update URL hash
                history.pushState(null, null, '#' + slug);
            }});
        }});
        
        // Handle URL hash on load
        function handleHash() {{
            const hash = window.location.hash.slice(1);
            if (hash && pages[hash]) {{
                const navItem = document.querySelector(`[data-slug="${{hash}}"]`);
                if (navItem) {{
                    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                    navItem.classList.add('active');
                }}
                loadMarkdown(hash);
            }}
        }}
        
        window.addEventListener('hashchange', handleHash);
        handleHash();
    </script>
</body>
</html>'''

# Write index.html
output_path = '2026-04-22-120026/index.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

file_size = os.path.getsize(output_path) / 1024
print(f'✅ index.html 生成成功: {output_path}')
print(f'   文件大小: {file_size:.1f} KB')
print(f'   文档页面: {len(pages_data)} 篇')

if file_size > 5000:
    print(f'   ⚠️  警告: 文件较大 ({file_size:.0f} KB)，建议启用 gzip 压缩')
