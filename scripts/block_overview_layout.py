"""Block页面的宏观入口与按节点展开；原必要讲解全文保留。"""
import re


def arrange_block(book, math):
    meta = book['lessons']['t_block']
    pieces = re.split(r'<h3>(.*?)</h3>', meta['body'])
    sections = [(pieces[i], pieces[i+1]) for i in range(1, len(pieces), 2)]
    assert len(sections) == 9, 'Block讲解结构变化时须核对折叠分组'
    m = lambda s: math(s, inline=True)
    full = lambda i: '<h3>'+sections[i][0]+'</h3>'+sections[i][1]
    summary = sections[-1][1]
    formula = re.search(r'<div class="math-block".*?</div>', summary, re.S).group()
    fold = lambda title, content, cls='': '<details class="block-detail '+cls+'"><summary>'+title+'</summary><div class="block-detail-body">'+content+'</div></details>'
    recap = '<p><code>MLPBranch</code> 是第四部分“基础层综合”的实现任务：将归一化与特征加工串联，残差加法留在本题完成。</p>'+math(r'\operatorname{MLPBranch}(U)=\operatorname{MLP}(\operatorname{RMSNorm}(U))')+'<button type="button" data-block-task="foundation_capstone">查看 MLP 分支前置任务</button>'
    why = fold('设计补充 · 为什么这里是两次残差？', '<p>本项目采用的 GPT Block 顺序完成两种工作：注意力读取上下文，随后 MLP 加工更新后的特征。因此设置两次残差，每次都保留该阶段输入，再加上分支补充。这是选定的模型结构；数学上也可以设计更多分支或其他结构。</p>', 'block-design-note')
    route = '<div class="block-route" aria-label="Block整体主线">'
    route += '<div class="block-route-state">输入已有表示 '+m('X:(B,T,C)')+'</div>'
    route += fold('<span class="block-node-index">01</span><span><strong>读取上下文，再加回原表示</strong><small>归一化 → 多头注意力 → 残差相加</small></span>', full(1)+full(2), 'block-route-node')
    route += '<div class="block-route-state">↓ 更新后的表示 '+m('X_1:(B,T,C)')+'</div>'
    route += fold('<span class="block-node-index">02</span><span><strong>加工更新后的特征，再补充一次</strong><small><code>MLP</code> 分支（归一化 → <code>MLP</code>）→ 残差相加</small></span>', recap+full(3), 'block-route-node')
    route += '<div class="block-route-state">↓ 输出新表示 '+m('Y:(B,T,C)')+'，交给后续 <code>Block</code> 或模型下一层</div></div>'
    before = '<h4>复用什么，本次写什么</h4><p>下面列出本题直接调用的三个前置任务。测评状态与代码来源分别显示，便于回到原题核对。</p><div data-block-prerequisites></div><p><strong>本次由你实现：</strong><code>Block.__init__</code> 创建并保留子层；<code>forward</code> 完成两次残差；<code>backward</code> 按逆序传递并合流。包装与组合代码也由你编写。</p>'
    forward_part = '<section class="block-learning-part"><h3>前向主线 · 信息怎样更新</h3>'+route+'</section>'
    backward_part = '<section class="block-learning-part"><h3>反向主线 · 梯度怎样返回</h3><p>沿上面的两步逆序返回：先完成第二次残差的合流，再回到第一次残差。</p>'+fold('展开完整反向：先看对应前向，再看两条路径',full(5)+full(6)+full(7))+'</section>'
    implementation = '<section class="block-learning-part"><h3>实现准备 · 对象怎样组织</h3><p>构造时创建子层，前后向复用同一批对象和缓存。</p>'+fold('构造与缓存：子层与参数怎样组织',full(4))+before+'</section>'
    meta['body'] = '<div class="block-macro"><h3>先看整体：这个 Block 完成什么</h3>'+sections[0][1]+'<p><strong>主线只有两步：读取上下文，再加工特征；每步都把补充加回当前表示。</strong></p><h3>核心公式 · 始终展开</h3>'+formula+'<p>两个归一化层都位于计算分支内部；直连保留各阶段的输入。</p>'+why+forward_part+backward_part+implementation+'<h3>'+sections[-1][0]+'</h3>'+summary.replace(formula,'').replace('整个单元可以记成：','按上方核心公式，可以对应为：')+'</div>'
    meta['body'] = meta['body'].replace('刚完成的', '前一节的').replace('已完成的 <code>MLPBranch</code>', '前置练习中的 <code>MLPBranch</code>').replace('复用第四部分完成的归一化与特征加工分支', '调用第四部分前置任务中的个人归一化与特征加工分支').replace('计算细节沿用已完成的个人模块', '计算细节调用前置任务中的个人模块')
    return book
